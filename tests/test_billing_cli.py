"""No replica: exercise payment boundaries and actual measurement dispatch."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import billing_cli
import measure_verdict
import profile_query

QUOTE = '(variant { Ok = record { required_attachment = 120_015_000_000 : nat } })'


class BillingCliTests(unittest.TestCase):
    def args(self, **kw):
        return SimpleNamespace(**(dict(proxy='proxy-id', max_cycles=120_015_000_000,
                                      execution_pricing=None, replica='http://localhost:8000') | kw))

    def test_exact_quote_and_no_silent_fee_increase(self):
        self.assertEqual(billing_cli.proxy_flags(self.args(), QUOTE),
                         ['--proxy', 'proxy-id', '--cycles', '120015000000'])
        for args, quote in [(self.args(max_cycles=1), QUOTE), (self.args(proxy=None), QUOTE),
                            (self.args(), '(variant { Err = Budget })'),
                            (self.args(), QUOTE.replace('120_015_000_000', '0'))]:
            with self.assertRaises(ValueError): billing_cli.proxy_flags(args, quote)

    def test_invalid_tariffs_and_cycle_amounts_are_rejected(self):
        for value in ['0', '-1', str(2**128)]:
            with self.assertRaises(ValueError): billing_cli.positive_u128(value)
        for value in ['1,0,1', '1,1', f'{2**64},1,1']:
            with self.assertRaises(ValueError): billing_cli.pricing(value)
        self.assertIn('instruction_cycles_denominator=13:nat64', billing_cli.pricing_arg('5000000,34,13'))

    def test_measurement_passes_payment_options_only_to_paid_inference(self):
        commands = []
        def run(command, **kw):
            commands.append(command)
            return SimpleNamespace(returncode=0, stdout='MEASURED_INSTRUCTIONS 123 tokens=3\n', stderr='')
        with patch.object(measure_verdict.subprocess, 'run', run):
            for query in [False, True]:
                measure_verdict.infer(None, self.args(), 'target', 'owner', Path('owner.pem'), [1,3,2], query=query)
        self.assertIn('--proxy', commands[0]); self.assertIn('--max-cycles', commands[0])
        self.assertNotIn('--proxy', commands[1]); self.assertNotIn('--execution-pricing', commands[1])

    def test_profile_update_quotes_then_proxies_without_proxying_queries_or_admin(self):
        commands = []
        def run(command, **kw):
            commands.append(command)
            if command[1:3] == ['network', 'status']:
                out = '{"managed":true,"api_url":"http://localhost:8000"}'
            elif command[1:3] == ['canister', 'status']:
                out = '{"module_hash":"test"}'
            else:
                method = command[4]
                out = {'query_limits': '(record { cost_fixed = 1; cost_per_token = 2 })',
                       'info': '(record { budget = 40000000000 })',
                       'cycles_pricing': QUOTE,
                       'infer_tokens': '(variant { Ok = record { measured_instructions = 123 } })',
                       'set_execution_pricing': '(variant { Ok })'}[method]
            return SimpleNamespace(returncode=0, stdout=out, stderr='')
        with tempfile.TemporaryDirectory() as d:
            argv = ['profile_query.py', '--canister', 'target', '--identity', 'owner', '--mode', 'update',
                    '--lengths', '6', '--repeats', '1', '--out', str(Path(d)/'result.json'),
                    '--proxy', 'proxy-id', '--max-cycles', '120015000000', '--execution-pricing', '5000000,1,1']
            with patch.object(sys, 'argv', argv), patch.object(profile_query.subprocess, 'run', run), contextlib.redirect_stdout(io.StringIO()):
                profile_query.main()
        calls = [c for c in commands if c[1:3] == ['canister', 'call']]
        paid = next(c for c in calls if c[4] == 'infer_tokens')
        self.assertEqual(paid[-4:], ['--proxy', 'proxy-id', '--cycles', '120015000000'])
        quote = next(c for c in calls if c[4] == 'cycles_pricing')
        self.assertIn('--query', quote)
        for c in calls:
            if c[4] != 'infer_tokens': self.assertNotIn('--proxy', c)

    def test_missing_payment_options_fail_before_any_rpc(self):
        with patch.object(sys, 'argv', ['profile_query.py', '--canister', 'target', '--identity', 'owner',
                                      '--mode', 'update', '--out', '/tmp/unused-billing-test.json']), \
             patch.object(profile_query.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, '--proxy'): profile_query.main()
            run.assert_not_called()


if __name__ == '__main__': unittest.main()
