"""Assistant-only alignment, reversible interventions, and cached suffix parity."""

import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_llm_sensitivity import (activation_qdq, aggregate, intervention,
    response_positions, score_batch, selected_log_probs, targets_for_model)
from int4 import PackedLayer
from report_llm_sensitivity import paired_groups


class MemoryCheckpoint:
    def __init__(self, name, module):
        packed = PackedLayer(module, "linear", 128, "torch", True)
        self.state = {f"{name}.{key}": value for key, value in packed.state_dict().items()}

    def get_tensor(self, name):
        return self.state[name]


class SensitivityTests(unittest.TestCase):
    def test_paired_group_interval_preserves_shared_example_variation(self):
        samples=[{'id':str(i),'fold':'discovery' if i%2==0 else 'confirmation'} for i in range(4)]
        rows=[]
        for mode,delta in (('w4a16',2),('w16a4',-1)):
            for block in range(2):
                for family in ('attention','ffn'):
                    rows.append({'mode':mode,'target':{'level':'group','family':family},
                        'examples':[{'id':str(i),'kl':2+i+block+(delta if family=='ffn' else 0)} for i in range(4)]})
        result=paired_groups(rows,samples)
        for mode,delta in (('w4a16',2),('w16a4',-1)):
            for fold in ('all','discovery','confirmation'):
                self.assertEqual(result[mode][fold]['mean_ffn_minus_attention_kl'],delta)
                self.assertEqual(result[mode][fold]['paired_bootstrap_95_ci'],[delta,delta])

    def test_only_assistant_next_token_positions_and_even_sampling(self):
        labels = torch.tensor([[-100,-100,5,6,-100,8],[-100,2,-100,-100,9,-100]])
        positions, targets, counts = response_positions(labels, 2)
        self.assertEqual(positions.tolist(), [1,4,6,9])
        self.assertEqual(targets.tolist(), [5,8,2,9])
        self.assertEqual(counts, [2,2])
        with self.assertRaisesRegex(ValueError,"assistant"):
            response_positions(torch.full((1,3),-100),32)

    def test_kl_direction_nll_and_example_macro_averaging(self):
        reference = torch.tensor([[.9,.1],[.5,.5],[.2,.8],[.4,.6]]).log()
        candidate = torch.tensor([[.7,.3],[.6,.4],[.1,.9],[.5,.5]]).log()
        batch = {"reference": reference, "targets": torch.tensor([0,1,1,0]),
                 "samples": [{"id":"a","fold":"discovery"},{"id":"b","fold":"confirmation"}],
                 "counts":[1,3]}
        rows,_ = score_batch(candidate,batch)
        expected = .9*math.log(.9/.7)+.1*math.log(.1/.3)
        self.assertAlmostEqual(rows[0]['kl'],expected,places=6)
        self.assertAlmostEqual(rows[0]['nll_delta'],math.log(.9/.7),places=6)
        metrics = aggregate(rows)
        self.assertAlmostEqual(metrics['all']['kl'],(rows[0]['kl']+rows[1]['kl'])/2,places=8)
        self.assertEqual(metrics['all']['tokens'],4)
        self.assertEqual(score_batch(reference,batch)[0][0]['kl'],0)

    def test_intervention_restores_parameters_and_hooks_after_failure(self):
        torch.manual_seed(3)
        model = nn.Module()
        model.projection = nn.Linear(128,7,bias=True)
        checkpoint = MemoryCheckpoint('projection',model.projection)
        before = {name: p.detach().clone() for name,p in model.named_parameters()}
        target = {'name':'projection','modules':['projection'],'kind':'linear'}
        x=torch.randn(2,128)
        for mode in ('w4a16','w16a4','w4a4'):
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                with intervention(model,target,mode,checkpoint):
                    if mode != 'w16a4':
                        self.assertFalse(torch.equal(model.projection.weight,before['projection.weight']))
                    self.assertTrue(torch.isfinite(model.projection(x)).all())
                    raise RuntimeError('deliberate')
            for name,p in model.named_parameters():
                self.assertTrue(torch.equal(p,before[name]))
                self.assertIsNone(p.grad)
            self.assertFalse(model.projection._forward_pre_hooks)

    def test_layer_coverage_groups_and_non_llm_exclusion(self):
        config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                           num_attention_heads=2,num_key_value_heads=2)
        model=LlamaForCausalLM(config)
        model.model.vision_tower=nn.Linear(8,8)
        targets=targets_for_model(model)
        self.assertEqual(sum(t['level']=='leaf' for t in targets),21)
        self.assertEqual(sum(t['level']=='group' for t in targets),4)
        self.assertFalse(any('vision' in t['name'] for t in targets))
        groups=[t for t in targets if t['level']=='group']
        self.assertEqual([len(t['modules']) for t in groups],[4,3,4,3])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_activation_values_match_native_packing_exactly(self):
        from int4_cuda import quantize_activation
        for k in (128,2048,5632,80):
            group=math.gcd(k,128)
            physical=(group+31)//32*32
            torch.manual_seed(k)
            x=torch.randn(9,k,device='cuda',dtype=torch.float16)
            x[0].zero_()
            x[1,:group]=torch.linspace(-1e-6,1e-6,group,device='cuda').half()
            x[2,0]=1000
            packed,scales=quantize_activation(x,group,physical)
            values=torch.stack((packed & 15,packed >> 4),-1).reshape(packed.shape[0],-1)
            values=values.to(torch.int8)
            values=torch.where(values>=8,values-16,values).reshape(packed.shape[0],-1,physical)
            expected=(values[:len(x),:,:group].float()*scales[:len(x),:,None].float()).half().reshape(x.shape)
            torch.testing.assert_close(activation_qdq(x),expected,atol=0,rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cached_prefix_final_norm_and_head_are_exact_and_restore_on_error(self):
        config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=8,
                           num_attention_heads=2,num_key_value_heads=2,_attn_implementation='sdpa')
        model=LlamaForCausalLM(config).cuda().half().eval().requires_grad_(False)
        labels=torch.tensor([[-100,-100,3,4,5,6,7,8,9],[-100,-100,3,4,5,6,-100,-100,-100]])
        rows,values,counts=response_positions(labels)
        batch={'embeds':torch.randn(2,9,16,dtype=torch.float16),'positions':None,
               'mask':torch.tensor([[True]*9,[True]*6+[False]*3]),'selected_rows':rows,
               'prefix_hidden':{}}
        handles=[]
        for index,module in ((4,model.model.layers[4]),(8,model.model.norm)):
            def capture(_module,args,key=index):
                batch['prefix_hidden'][key]=args[0].detach().cpu().clone()
            handles.append(module.register_forward_pre_hook(capture))
        baseline=selected_log_probs(model,batch,capture=True)
        for handle in handles: handle.remove()
        for name,block in (('model.layers.4.self_attn.q_proj',4),('model.norm',None),('lm_head',None)):
            candidate=selected_log_probs(model,batch,{'name':name,'block':block,'kind':'linear'})
            torch.testing.assert_close(candidate,baseline,atol=0,rtol=0)
        layers=model.model.layers
        with patch.object(layers[4],'forward',side_effect=RuntimeError('deliberate')):
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                selected_log_probs(model,batch,{'name':'model.layers.4','block':4,'kind':'linear'})
        self.assertIs(model.model.layers,layers)
        self.assertEqual(len(layers),8)
        # Future right padding must not affect the scored causal prefix.
        with torch.inference_mode():
            unmasked=model.model(inputs_embeds=batch['embeds'].cuda(),attention_mask=None,use_cache=False).last_hidden_state
            masked=model.model(inputs_embeds=batch['embeds'].cuda(),attention_mask=batch['mask'].cuda(),use_cache=False).last_hidden_state
        torch.testing.assert_close(unmasked[1,:6],masked[1,:6],atol=0.004,rtol=0.005)


if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
