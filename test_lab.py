import math
import unittest
import torch
from catalog import PROBLEMS
from runner import evaluate, compare


class LabTests(unittest.TestCase):
    def test_parameter_documentation(self):
        import inspect
        for slug,p in PROBLEMS.items():
            with self.subTest(problem=slug):
                ns={};exec(p['reference'],ns)
                self.assertEqual(list(inspect.signature(ns['solve']).parameters),[v['name'] for v in p['inputs']])
                self.assertTrue(p['returns'])
                for field in p['inputs']+p['returns']:
                    self.assertTrue(all(field[k] for k in ('name','type','shape','description')))

    def test_exported_reference_modules_and_tensor_tutorial(self):
        from pathlib import Path
        from tensor_guide import TENSOR_GUIDE
        for slug in PROBLEMS:
            with self.subTest(problem=slug):
                code=(Path(__file__).parent/'references'/(slug+'.py')).read_text(encoding='utf-8')
                result=evaluate(dict(problem=slug,code=code,mode='submit'))
                self.assertTrue(result['passed'],result)
        exec(TENSOR_GUIDE, {})

    def test_all_references_and_unimplemented(self):
        for slug,p in PROBLEMS.items():
            with self.subTest(problem=slug):
                result=evaluate(dict(problem=slug,code=p['reference'],mode='submit'))
                errors=[c.get('error') for c in result['cases'] if not c['passed']]
                self.assertTrue(result['passed'],str(result.get('error') or errors))
                self.assertFalse(evaluate(dict(problem=slug,code=p['starter'],mode='run'))['passed'])

    def test_rl_scalar_oracles(self):
        # Deliberately unequal sequence lengths, both signs, both clipping sides.
        new=torch.tensor([[0.4,-0.8,0.1],[-0.6,0.7,-0.1]],dtype=torch.float64)
        old=torch.zeros_like(new); mask=torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool)
        advantage=torch.tensor([1.5,-0.7],dtype=torch.float64)
        per_seq=[]; token_sum=0.; count=0; gspo=[]
        for i in range(2):
            values=[]; logs=[]
            for j in range(3):
                if mask[i,j]:
                    r=math.exp(float(new[i,j])); a=float(advantage[i]); logs.append(float(new[i,j]))
                    values.append(min(r*a,min(1.28,max(0.8,r))*a))
            per_seq.append(sum(values)/len(values)); token_sum+=sum(values); count+=len(values)
            ratio=math.exp(sum(logs)/len(logs)); a=float(advantage[i])
            gspo.append(min(ratio*a,min(1.28,max(0.8,ratio))*a))
        targets={'grpo_loss':-sum(per_seq)/2,'dapo_loss':-token_sum/count,'gspo_loss':-sum(gspo)/2}
        self.assertEqual(len(set(round(v,6) for v in targets.values())),3)
        for slug,expected in targets.items():
            ns={};exec(PROBLEMS[slug]['reference'],ns)
            self.assertAlmostEqual(float(ns['solve'](new,old,advantage,mask)),expected,places=10)

    def test_wrong_implementations_rejected(self):
        mutations={
            'ppo_loss':PROBLEMS['ppo_loss']['reference'].replace('torch.minimum','torch.maximum'),
            'gspo_loss':PROBLEMS['dapo_loss']['reference'],
            'dapo_loss':PROBLEMS['grpo_loss']['reference'],
            'rmsnorm':'import torch\ndef solve(x,weight,eps=1e-6):\n return (x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*weight).detach()',
            'softmax':'import torch\ndef solve(x,dim=-1):\n return x.exp()/x.exp().sum(dim,keepdim=True)',
            'gae_advantage':PROBLEMS['gae_advantage']['reference'].replace('live=(~terminated[t]).to(rewards.dtype)','live=torch.ones_like(rewards[t])'),
        }
        for slug,code in mutations.items():
            with self.subTest(problem=slug):
                self.assertFalse(evaluate(dict(problem=slug,code=code,mode='submit'))['passed'])
        zero_grad='import torch\ndef solve(x,dim=-1):\n return torch.softmax(x,dim).detach()+x*0'
        self.assertFalse(evaluate(dict(problem='softmax',code=zero_grad,mode='submit'))['passed'])

    def test_independent_torch_operators(self):
        functions={
            'softmax':lambda x,dim=-1: torch.softmax(x,dim),
            'cross_entropy':torch.nn.functional.cross_entropy,
            'layernorm':lambda x,w,b,eps=1e-5:torch.nn.functional.layer_norm(x,(x.shape[-1],),w,b,eps),
            'conv2d':torch.nn.functional.conv2d,
            'max_pool2d':torch.nn.functional.max_pool2d,
            'depthwise_conv':lambda x,d,p:torch.nn.functional.conv2d(torch.nn.functional.conv2d(x,d,padding=d.shape[-1]//2,groups=x.shape[1]),p),
            'flash_attention':lambda q,k,v,block_size=2:torch.nn.functional.scaled_dot_product_attention(q,k,v),
            'ring_attention':lambda q,k,v,block_size=2:torch.nn.functional.scaled_dot_product_attention(q,k,v),
        }
        for slug,fn in functions.items():
            for case in PROBLEMS[slug]['cases']:
                with self.subTest(problem=slug,case=case['name']):
                    torch.manual_seed(53); fixture={'torch':torch};exec(case['setup'],fixture)
                    ns={};exec(PROBLEMS[slug]['reference'],ns)
                    compare(ns['solve'](*fixture['args']),fn(*fixture['args']))

    def test_debug_and_trace(self):
        code='import torch\ndef solve(x):\n    y=x*2\n    return y\n'
        r=evaluate(dict(problem='softmax',code=code,mode='debug',custom='print(solve(torch.tensor([3.])))',breakpoints=[4]))
        self.assertTrue(r['passed']); self.assertIn('6.',r['stdout'])
        self.assertEqual(r['snapshots'][0]['line'],4); self.assertIn('y',r['snapshots'][0]['locals'])

    def test_einops_in_submission_and_debug(self):
        code = '''import math
import torch
from einops import rearrange

def solve(x, wq, wk, wv, wo, heads):
    q,k,v=[rearrange(x@w,'b t (h d) -> b h t d',h=heads) for w in (wq,wk,wv)]
    y=(q@k.transpose(-2,-1)/math.sqrt(q.shape[-1])).softmax(-1)@v
    return rearrange(y,'b h t d -> b t (h d)')@wo
'''
        result=evaluate(dict(problem='mha',code=code,mode='submit'))
        self.assertTrue(result['passed'],result)
        debug=evaluate(dict(problem='mha',code=code,mode='debug',custom="from einops import repeat, reduce\nx=repeat(torch.arange(3.), 'd -> b d', b=2)\nprint(reduce(x, 'b d -> d', 'mean'))"))
        self.assertTrue(debug['passed'],debug)
        self.assertIn('tensor',debug['stdout'])

    def test_compile_error(self):
        r=evaluate(dict(problem='softmax',code='def solve(:',mode='run'))
        self.assertFalse(r['passed']);self.assertIn('SyntaxError',r['error'])


if __name__=='__main__': unittest.main(verbosity=2)
