import unittest
from collections import Counter
from semantic_game_study import DATA, LABELS, fixtures, summarize

class StudyTest(unittest.TestCase):
    def test_balanced_cases_and_changed_pairs(self):
        for game, cases in DATA.items():
            self.assertEqual(len(cases),24)
            self.assertEqual(set(Counter(c[0] for c in cases).values()),{24//len(LABELS[game])})
            for i in range(0,24,2):
                self.assertNotEqual(cases[i][0],cases[i+1][0])
    def test_fixtures(self):
        rows=fixtures()
        self.assertEqual(len(rows),288)
        self.assertEqual(len({r['id'] for r in rows}),288)
        self.assertEqual(len({(r['text'],tuple(r['labels'])) for r in rows}),288)
        for row in rows:
            self.assertIn(row['expected'],row['labels'])
            self.assertEqual(set(row['labels']),set(LABELS[row['game']]))
    def test_unanswered_is_not_consistent(self):
        for result in summarize(fixtures()).values():
            self.assertEqual(result['completed'],0)
            self.assertEqual(result['order_consistent'],0)

    def test_perfect_and_constant(self):
        perfect=[dict(r,answer=r['expected']) for r in fixtures()]
        for s in summarize(perfect).values():
            self.assertEqual(s['correct'],96)
            self.assertEqual(s['all_eight_pair_variants_correct'],12)
            self.assertEqual(s['order_consistent'],48)
        constant=[dict(r,answer=LABELS[r['game']][0]) for r in fixtures()]
        for game,s in summarize(constant).items():
            self.assertEqual(s['correct'],96//len(LABELS[game]))
            self.assertEqual(s['all_eight_pair_variants_correct'],0)

if __name__=='__main__':unittest.main()
