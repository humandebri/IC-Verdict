import unittest
from game_feasibility import slide, snake_options, spawn
import random

class RulesTest(unittest.TestCase):
    def test_merge_once_and_score(self):
        board = [2,2,2,2]+[0]*12
        after,score = slide(board,3)
        self.assertEqual(after[:4],[4,4,0,0]); self.assertEqual(score,8)
        self.assertEqual(board[:4],[2,2,2,2])
        self.assertEqual(slide([2,2,4,0]+[0]*12,3)[0][:4],[4,4,0,0])
    def test_directions(self):
        board = [2,0,0,0,2,0,0,0]+[0]*8
        for action,index in [(0,0),(2,12)]:
            after,score = slide(board,action)
            self.assertEqual(after[index],4); self.assertEqual(score,4)
            self.assertEqual(sum(after),4)
        self.assertEqual(slide([2,2,0,0]+[0]*12,1)[0][:4],[0,0,0,4])
    def test_spawn(self):
        board = [0]*16
        spawn(board,random.Random(11))
        self.assertEqual(sum(x!=0 for x in board),1)
        self.assertIn(max(board),(2,4))
    def test_snake_tail_vacates_but_body_kills(self):
        body = [(1,1),(1,2),(0,2),(0,1)]
        opts = snake_options(body,(5,5),0)
        self.assertNotIn(2,opts)
        self.assertFalse(opts[3]['death'])
        opts = snake_options([(0,0),(0,1),(1,1),(1,0),(2,0)],(5,5),0)
        self.assertTrue(opts[0]['death']); self.assertTrue(opts[1]['death'])
    def test_snake_eating(self):
        opts=snake_options([(2,3),(1,3),(0,3)],(3,3),1)
        self.assertTrue(opts[1]['eating']); self.assertFalse(opts[1]['death'])

if __name__=='__main__': unittest.main()
