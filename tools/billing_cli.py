"""Shared argument validation for explicit, capped icp proxy payments."""
import re

PAID_METHODS = {'infer_tokens', 'decide', 'decide_batch', 'evaluate'}


def positive_u128(value):
    number = int(value)
    if not 0 < number < 2**128:
        raise ValueError('cycles must be a positive u128')
    return number


def pricing(value):
    parts = [int(x) for x in value.split(',')]
    if len(parts) != 3 or any(not 0 < x < 2**64 for x in parts):
        raise ValueError('pricing requires positive u64 base,numerator,denominator')
    return value


def add_arguments(parser):
    parser.add_argument('--proxy', help='funded icp proxy authorized for the calling identity')
    parser.add_argument('--max-cycles', type=positive_u128, help='maximum attachment per paid call (decimal cycles)')
    parser.add_argument('--execution-pricing', type=pricing, help='owner-only tariff setup: base,numerator,denominator')


def require_payment(args):
    if not args.proxy or not args.max_cycles:
        raise ValueError('paid inference requires --proxy PRINCIPAL --max-cycles N')


def pricing_arg(value):
    base, numerator, denominator = map(int, pricing(value).split(','))
    return f'(opt record {{base_cycles={base}:nat64;instruction_cycles_numerator={numerator}:nat64;instruction_cycles_denominator={denominator}:nat64}})'


def proxy_flags(args, quote):
    require_payment(args)
    if 'Err' in quote:
        raise ValueError('cycles_pricing rejected; configure execution pricing first: ' + quote)
    match = re.search(r'\brequired_attachment\s*=\s*([\d_]+)', quote)
    if not match:
        raise ValueError('cycles_pricing response lacks required_attachment')
    required = int(match[1].replace('_', ''))
    if not 0 < required <= args.max_cycles:
        raise ValueError(f'required attachment {required} exceeds --max-cycles {args.max_cycles} or is zero')
    return ['--proxy', args.proxy, '--cycles', str(required)]


def uploader_flags(args, *, paid=True):
    flags = []
    if paid:
        require_payment(args)
        flags = ['--proxy', args.proxy, '--max-cycles', str(args.max_cycles)]
    if getattr(args, 'execution_pricing', None):
        flags += ['--execution-pricing', args.execution_pricing]
    return flags
