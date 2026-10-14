import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from life.core import Config, make_grid, reference_step, validate_grid
from life.cpu import CPUEngine
from life.gpu import CUDAEngine, cuda_status, measure_cuda
from life.benchmark import benchmark, suite, save_report, write_csv

class RulesTests(unittest.TestCase):
    def test_all_neighbor_counts_for_live_and_dead_cells(self):
        neighbours = [(r,c) for r in range(3) for c in range(3) if (r,c)!=(1,1)]
        for alive in (0,1):
            for count in range(9):
                a=np.zeros((3,3),dtype=np.uint8);a[1,1]=alive
                for r,c in neighbours[:count]: a[r,c]=1
                expected=int(count==3 or (alive and count==2))
                self.assertEqual(int(CPUEngine(a,'dead').step()[1,1]),expected)
                self.assertEqual(int(reference_step(a,'dead')[1,1]),expected)

    def test_block_is_stable(self):
        a=make_grid(Config(size=8,pattern='block'))
        np.testing.assert_array_equal(CPUEngine(a).advance(100),a)

    def test_blinker_period_two(self):
        a=make_grid(Config(size=8,pattern='blinker'))
        engine=CPUEngine(a)
        self.assertFalse(np.array_equal(engine.step(),a))
        np.testing.assert_array_equal(engine.step(),a)

    def test_glider_translates_after_four_generations(self):
        a=make_grid(Config(size=16,pattern='glider'))
        np.testing.assert_array_equal(CPUEngine(a).advance(4),np.roll(a,(1,1),(0,1)))

    def test_boundaries_differ(self):
        a=np.zeros((5,5),dtype=np.uint8);a[0,[0,1,4]]=1
        wrap=CPUEngine(a,'wrap').step();dead=CPUEngine(a,'dead').step()
        self.assertEqual(wrap[4,0],1);self.assertEqual(dead[4,0],0)

    def test_random_rectangles_match_scalar_reference(self):
        for shape in ((1,1),(1,7),(2,3),(7,11),(16,17)):
            for boundary in ('wrap','dead'):
                a=(np.random.default_rng(5).random(shape)<.4).astype(np.uint8)
                engine=CPUEngine(a,boundary)
                for _ in range(6):
                    a=reference_step(a,boundary)
                    np.testing.assert_array_equal(engine.step(),a)

    def test_input_not_mutated(self):
        a=make_grid(Config(size=8));copy=a.copy();CPUEngine(a).advance(5)
        np.testing.assert_array_equal(a,copy)

    def test_zero_iterations_preserves_grid(self):
        a=make_grid(Config(size=8));np.testing.assert_array_equal(CPUEngine(a).advance(0),a)

    def test_seed_is_reproducible(self):
        np.testing.assert_array_equal(make_grid(Config()),make_grid(Config()))
        self.assertFalse(np.array_equal(make_grid(Config(seed=1)),make_grid(Config(seed=2))))

    def test_patterns_fit_and_have_expected_population(self):
        for name,count in [('glider',5),('blinker',3),('block',4),('r-pentomino',5),('gosper-gun',36),('empty',0)]:
            self.assertEqual(int(make_grid(Config(size=64,pattern=name)).sum()),count)
        with self.assertRaises(ValueError):make_grid(Config(size=8,pattern='gosper-gun'))

    def test_invalid_inputs(self):
        for kwargs in ({'size':0},{'size':True},{'size':4096},{'iterations':0},{'repeats':11},
                       {'seed':-1},{'density':float('nan')},{'boundary':'bad'},{'block_x':1024}):
            with self.assertRaises(ValueError):Config(**kwargs)
        for a in ([[2]],[],[1,0],[[float('nan')]]):
            with self.assertRaises(ValueError):validate_grid(a)
        with self.assertRaises(ValueError):CPUEngine([[0]],'invalid')
        with self.assertRaises(ValueError):CPUEngine([[0]]).advance(-1)

class BenchmarkTests(unittest.TestCase):
    def test_cpu_samples_and_exports(self):
        report=suite([Config(size=8,iterations=3,repeats=2)],mode='cpu')
        row=report['results'][0]
        self.assertEqual(len(row['cpu']['samples']),2)
        self.assertIsNone(row['cuda']);self.assertIsNone(row['verified'])
        self.assertGreater(row['cpu']['total_ms'],row['cpu']['compute_ms'])
        with tempfile.TemporaryDirectory() as path:
            save_report(report,path)
            with open(path+'/benchmark.json') as f:self.assertEqual(json.load(f),report)
        stream=io.StringIO();write_csv(report,stream)
        self.assertIn('cuda_compute_ms',stream.getvalue())

    @patch('life.benchmark.cuda_status',return_value={'available':False,'simulator':False,'reason':'No GPU'})
    def test_missing_cuda_never_fakes_timing(self,_):
        result=benchmark(Config(size=8,iterations=2,repeats=1))
        self.assertIsNone(result['cuda']);self.assertIsNone(result['speedup_total'])
        self.assertEqual(result['note'],'No GPU')
        with self.assertRaises(RuntimeError):benchmark(Config(size=8),mode='cuda')

    @patch('life.benchmark.cuda_status',return_value={'available':True,'simulator':False})
    @patch('life.benchmark.warmup_cuda')
    @patch('life.benchmark.measure_cuda')
    def test_mismatch_withholds_speedup(self,measure,warmup,status):
        config=Config(size=8,iterations=1,repeats=1,pattern='block')
        measure.return_value=(np.zeros((8,8),dtype=np.uint8),{'compute_ms':1,'total_ms':2})
        result=benchmark(config)
        self.assertIs(result['verified'],False);self.assertIsNone(result['speedup_total'])

    @patch('life.benchmark.cuda_status',return_value={'available':True,'simulator':False})
    @patch('life.benchmark.warmup_cuda')
    @patch('life.benchmark.measure_cuda')
    def test_matching_results_compute_speedup(self,measure,warmup,status):
        config=Config(size=8,iterations=1,repeats=1,pattern='block')
        measure.return_value=(make_grid(config),{'compute_ms':1,'total_ms':2})
        result=benchmark(config)
        self.assertIs(result['verified'],True)
        self.assertEqual(result['speedup_total'],result['cpu']['total_ms']/2)

@unittest.skipUnless(cuda_status()['available'] or cuda_status()['simulator'], 'CUDA or simulator required')
class CUDATests(unittest.TestCase):
    def test_rectangular_partial_blocks_and_boundaries(self):
        for boundary in ('wrap','dead'):
            for shape in ((1,1),(2,3),(7,11),(17,19)):
                a=(np.random.default_rng(17).random(shape)<.35).astype(np.uint8)
                gpu=CUDAEngine(a,boundary,(8,8),allow_simulator=True)
                cpu=CPUEngine(a,boundary)
                for _ in range(3):
                    gpu.advance(1)
                    np.testing.assert_array_equal(gpu.result(),cpu.step())

    def test_all_block_shapes_and_double_buffer_parity(self):
        a=make_grid(Config(size=8,pattern='glider'))
        for block in ((8,8),(16,16),(32,8),(32,16),(32,32)):
            gpu=CUDAEngine(a,'wrap',block,allow_simulator=True)
            np.testing.assert_array_equal(gpu.result(),a)
            gpu.advance(4)
            np.testing.assert_array_equal(gpu.result(),CPUEngine(a).advance(4))

    def test_simulator_is_rejected_by_benchmark(self):
        if not cuda_status()['simulator']:self.skipTest('Only applies to simulator')
        with self.assertRaises(RuntimeError):measure_cuda(np.zeros((8,8),dtype=np.uint8),1,'wrap',(8,8))
        result=benchmark(Config(size=8,iterations=1,repeats=1))
        self.assertIsNone(result['cuda'])

if __name__=='__main__':unittest.main()
