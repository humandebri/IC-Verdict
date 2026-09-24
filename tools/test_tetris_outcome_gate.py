import copy
import unittest
from tetris_outcome_gate import cases, request, label, summarize

class OutcomeGateTests(unittest.TestCase):
    def test_balanced_unique_paired_questions(self):
        rows=cases()
        self.assertEqual(len(rows),100)
        self.assertEqual(len({r['id'] for r in rows}),100)
        self.assertEqual(len({str(request(r)) for r in rows}),100)
        self.assertEqual([sum(r['expected']==a for r in rows) for a in range(5)],[20]*5)
        for a,b in zip(rows[::2],rows[1::2]):
            self.assertEqual(a['pair'],b['pair'])
            self.assertNotEqual(a['expected'],b['expected'])
            swapped=copy.deepcopy(a['outcomes'])
            i,j=a['expected'],b['expected'];swapped[i],swapped[j]=swapped[j],swapped[i]
            self.assertEqual(swapped,b['outcomes'])

    def test_unique_dominating_outcome_and_no_answer_leak(self):
        for row in cases():
            winner=row['outcomes'][row['expected']]
            for a,outcome in enumerate(row['outcomes']):
                label(a,outcome)
                if a==row['expected']:continue
                self.assertGreaterEqual(winner['lines'],outcome['lines'])
                self.assertLessEqual(winner['holes'],outcome['holes'])
                self.assertLessEqual(winner['height'],outcome['height'])
                self.assertNotEqual(winner,outcome)
            changed={**row,'expected':(row['expected']+1)%5,'metric':'SECRET','id':'SECRET'}
            self.assertEqual(request(row),request(changed))

    def test_gate_enforces_action_coverage_and_swapped_pairs(self):
        rows=[dict(r,action=r['expected']) for r in cases()]
        self.assertTrue(summarize(rows)['passed'])
        affected=[r for r in rows if r['expected']==2][:5]
        for row in affected:row['action']=0
        result=summarize(rows)
        self.assertTrue(result['gates']['accuracy'])
        self.assertFalse(result['passed']);self.assertFalse(result['gates']['every_action'])
        self.assertFalse(summarize([dict(r,action=4) for r in cases()])['passed'])
        with self.assertRaises(ValueError):summarize(rows[:-1])
        changed=copy.deepcopy(rows);changed[0]['expected']=4
        with self.assertRaises(ValueError):summarize(changed)

    def test_outcome_bounds_and_topout(self):
        self.assertEqual(label(2,None),'rotate game over')
        for field,value in [('lines',5),('holes',201),('height',21),('height',-1),('lines',True)]:
            with self.assertRaises(ValueError):label(0,dict(lines=0,holes=0,height=0)|{field:value})

if __name__=='__main__':unittest.main()
