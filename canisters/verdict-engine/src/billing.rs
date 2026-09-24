//! Synchronous, caller-funded update execution. No ledger or async settlement.
use candid::CandidType;
use ic_laya_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::marker::PhantomData;

pub const MULTIPLIER: u64 = 3;

/// Actual subnet execution tariff, NOT the model's estimated instruction count.
/// The owner must configure this from the subnet's current IC tariff. Keeping it
/// optional makes installs/upgrades fail closed instead of assuming 13 nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, CandidType, Serialize, Deserialize)]
pub struct ExecutionPricing {
    pub base_cycles: u64,
    pub instruction_cycles_numerator: u64,
    pub instruction_cycles_denominator: u64,
}

impl ExecutionPricing {
    pub fn validate(self) -> Result<Self> {
        if self.base_cycles == 0 || self.instruction_cycles_numerator == 0
            || self.instruction_cycles_denominator == 0
        {
            return Err(Error::Invalid("execution pricing must be positive".into()));
        }
        Ok(self)
    }

    pub fn fee(self, instructions: u64) -> Result<u128> {
        self.validate()?;
        // u64 * u64 fits u128. Round the execution cost up to a whole cycle.
        let variable = (u128::from(instructions) * u128::from(self.instruction_cycles_numerator))
            .div_ceil(u128::from(self.instruction_cycles_denominator));
        variable.checked_add(u128::from(self.base_cycles))
            .and_then(|cost| cost.checked_mul(u128::from(MULTIPLIER)))
            .ok_or(Error::Budget)
    }

    pub fn require_attachment(self, available: u128) -> Result<()> {
        if available < self.fee(crate::UPDATE_BUDGET)? { Err(Error::Budget) } else { Ok(()) }
    }
}

thread_local! {
    static PRICING: RefCell<Option<ExecutionPricing>> = const { RefCell::new(None) };
}

pub fn get() -> Option<ExecutionPricing> { PRICING.with(|p| *p.borrow()) }
pub fn set(pricing: Option<ExecutionPricing>) { PRICING.with(|p| *p.borrow_mut() = pricing); }

#[derive(CandidType, Serialize, Deserialize)]
pub struct CyclesPricing {
    pub execution: ExecutionPricing,
    pub multiplier: u64,
    pub instruction_limit: u64,
    /// Attach this upper bound; only the measured fee is accepted.
    pub required_attachment: u128,
}

pub fn quote() -> Result<CyclesPricing> {
    let execution = get().ok_or_else(|| Error::Denied("execution pricing not configured".into()))?;
    Ok(CyclesPricing { execution, multiplier: MULTIPLIER, instruction_limit: crate::UPDATE_BUDGET,
        required_attachment: execution.fee(crate::UPDATE_BUDGET)? })
}

pub fn paid<T: CandidType>(work: impl FnOnce() -> Result<T>) -> PhantomData<Result<T>> {
    // No work (including cache mutation) starts until the full execution ceiling
    // is funded. There is no await, so attached cycles cannot change underneath us.
    let admitted = quote().and_then(|quote| {
        quote.execution.require_attachment(ic_cdk::api::msg_cycles_available())?;
        Ok(quote.execution)
    });
    let (pricing, result) = match admitted {
        Ok(pricing) => (Some(pricing), work()),
        Err(error) => (None, Err(error)),
    };
    let bytes = candid::encode_one(result).unwrap_or_else(|_| ic_cdk::trap("reply encoding failed"));
    // Includes Candid argument decoding, validation, postprocessing and reply
    // encoding. Only the small settlement/reply-copy tail is excluded.
    // Charge completed Result::Err calls too; a Wasm trap rolls back acceptance.
    if let Some(pricing) = pricing {
        let fee = pricing.fee(ic_cdk::api::instruction_counter())
            .unwrap_or_else(|_| ic_cdk::trap("execution fee overflow"));
        if ic_cdk::api::msg_cycles_accept(fee) != fee {
            ic_cdk::trap("insufficient cycles at settlement");
        }
    }
    ic_cdk::api::msg_reply(bytes);
    PhantomData
}

pub fn query_only() -> Result<()> {
    if ic_cdk::api::in_replicated_execution() {
        Err(Error::Denied("query-only endpoint; use a paid update for replicated inference".into()))
    } else { Ok(()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn tariff() -> ExecutionPricing {
        ExecutionPricing { base_cycles: 5_000_000, instruction_cycles_numerator: 1,
            instruction_cycles_denominator: 1 }
    }
    #[test]
    fn charges_three_times_execution_not_three_times_the_deposit() {
        let p = tariff();
        assert_eq!(p.fee(1_000_000_000).unwrap(), 3_015_000_000);
        let deposit = p.fee(crate::UPDATE_BUDGET).unwrap();
        assert_eq!(deposit, 120_015_000_000);
        assert!(p.require_attachment(deposit).is_ok());
        assert_eq!(p.require_attachment(deposit - 1), Err(Error::Budget));
        assert_eq!(p.require_attachment(0), Err(Error::Budget));
        assert_eq!(deposit - p.fee(1_000_000_000).unwrap(), 117_000_000_000);
    }
    #[test]
    fn fractional_tariff_and_large_instruction_counts_do_not_undercharge() {
        let p = ExecutionPricing { base_cycles: 13_076_923,
            instruction_cycles_numerator: 34, instruction_cycles_denominator: 13 };
        assert_eq!(p.fee(13).unwrap(), 3 * (13_076_923 + 34));
        assert_eq!(p.fee(1).unwrap(), 3 * (13_076_923 + 3));
        assert!(p.fee(u64::MAX).is_ok());
        let huge = ExecutionPricing { base_cycles: u64::MAX,
            instruction_cycles_numerator: u64::MAX, instruction_cycles_denominator: 1 };
        assert_eq!(huge.fee(u64::MAX), Err(Error::Budget));
    }
    #[test]
    fn zero_tariffs_are_rejected() {
        assert!(ExecutionPricing { base_cycles: 0, ..tariff() }.validate().is_err());
        assert!(ExecutionPricing { instruction_cycles_numerator: 0, ..tariff() }.validate().is_err());
        assert!(ExecutionPricing { instruction_cycles_denominator: 0, ..tariff() }.validate().is_err());
    }
}
