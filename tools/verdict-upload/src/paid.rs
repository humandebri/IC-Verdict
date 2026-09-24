//! icp-cli proxy protocol: the authorized proxy pays from its own balance.
//! https://github.com/dfinity/icp-cli/blob/main/crates/icp-canister-interfaces/src/proxy.rs
use candid::{CandidType, Decode, Encode, Nat, Principal};
use ic_agent::Agent;
use serde::Deserialize;

#[derive(CandidType, Deserialize)]
struct Quote { required_attachment: u128 }
#[derive(CandidType, Deserialize)]
struct ProxyArgs { canister_id: Principal, method: String, args: Vec<u8>, cycles: Nat }
#[derive(CandidType, Deserialize)]
struct ProxyOk { result: Vec<u8> }
#[derive(CandidType, Deserialize, Debug)]
enum ProxyError {
    InsufficientCycles { available: Nat, required: Nat },
    CallFailed { reason: String },
    UnauthorizedUser,
}

pub fn check_attachment(required: u128, maximum: u128) -> Result<(), String> {
    if required == 0 || required > maximum {
        return Err(format!("required attachment {required} exceeds --max-cycles {maximum} or is zero"));
    }
    Ok(())
}

pub async fn call(agent: &Agent, canister: Principal, proxy: Principal, maximum: u128,
                  method: &str, args: Vec<u8>) -> Result<Vec<u8>, String> {
    let raw = agent.query(&canister, "cycles_pricing")
        .with_arg(Encode!().map_err(|e| e.to_string())?).call().await
        .map_err(|e| format!("cycles_pricing: {e}"))?;
    let quote = Decode!(&raw, Result<Quote, ic_laya_core::Error>)
        .map_err(|e| format!("cycles_pricing reply: {e}"))?
        .map_err(|e| format!("cycles_pricing rejected: {e:?}; configure execution pricing first"))?;
    check_attachment(quote.required_attachment, maximum)?;
    let request = ProxyArgs { canister_id: canister, method: method.into(), args,
        cycles: quote.required_attachment.into() };
    let raw = agent.update(&proxy, "proxy").with_arg(Encode!(&request).map_err(|e| e.to_string())?)
        .call_and_wait().await.map_err(|e| format!("{method} proxy: {e}"))?;
    Decode!(&raw, Result<ProxyOk, ProxyError>).map_err(|e| format!("proxy reply: {e}"))?
        .map(|ok| ok.result).map_err(|e| format!("{method} proxy rejected: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quote_cannot_raise_the_operator_spending_limit() {
        assert!(check_attachment(120_015_000_000, 120_015_000_000).is_ok());
        assert!(check_attachment(120_015_000_001, 120_015_000_000).is_err());
        assert!(check_attachment(0, u128::MAX).is_err());
    }
    #[test]
    fn proxy_preserves_target_error_and_refuses_transport_errors() {
        let target = Encode!(&Err::<(), _>(ic_laya_core::Error::Budget)).unwrap();
        let raw = Encode!(&Ok::<_, ProxyError>(ProxyOk { result: target.clone() })).unwrap();
        assert_eq!(Decode!(&raw, Result<ProxyOk, ProxyError>).unwrap().unwrap().result, target);
        let raw = Encode!(&Err::<ProxyOk, _>(ProxyError::UnauthorizedUser)).unwrap();
        assert!(matches!(Decode!(&raw, Result<ProxyOk, ProxyError>).unwrap(), Err(ProxyError::UnauthorizedUser)));
    }
}
