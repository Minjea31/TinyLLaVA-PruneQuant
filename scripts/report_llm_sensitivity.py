"""Verify a completed sensitivity scan and summarize projections and paired groups."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import numpy as np

from common import ROOT, now, sha256, write_json
from analyze_llm_sensitivity import MODES, write_report


def paired_groups(rows, sample_manifest, seed=2026):
    """Bootstrap calibration examples; condition on the existing 22-block model."""
    output = {}
    for mode in MODES:
        by_family = {}
        for family in ("attention", "ffn"):
            selected = [r for r in rows if r['mode'] == mode and r['target']['level'] == 'group'
                        and r['target']['family'] == family]
            if not selected:
                break
            by_family[family] = {s['id']: statistics.mean(
                next(e['kl'] for e in r['examples'] if e['id'] == s['id']) for r in selected)
                for s in sample_manifest}
        if len(by_family) != 2:
            continue
        output[mode] = {}
        for fold in ('all','discovery','confirmation'):
            ids = [s['id'] for s in sample_manifest if fold == 'all' or s['fold'] == fold]
            if not ids:
                continue
            attention = np.array([by_family['attention'][key] for key in ids])
            ffn = np.array([by_family['ffn'][key] for key in ids])
            delta = ffn - attention
            rng = np.random.default_rng(seed)
            bootstrap = delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(axis=1)
            output[mode][fold] = {'examples':len(ids), 'mean_attention_kl':float(attention.mean()),
                'mean_ffn_kl':float(ffn.mean()), 'mean_ffn_minus_attention_kl':float(delta.mean()),
                'paired_bootstrap_95_ci':np.quantile(bootstrap,[.025,.975]).tolist()}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/sensitivity_llm')
    args=parser.parse_args()
    directory=args.output_dir
    spec=json.loads((directory/'run.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if summary['status'] != 'complete':
        raise RuntimeError('Sensitivity scan is not complete')
    for name,digest in spec['source_hashes'].items():
        if sha256(Path(name)) != digest:
            raise RuntimeError(f'Sensitivity source changed after the scan: {name}')
    rows=[json.loads(line) for line in (directory/'measurements.jsonl').read_text().splitlines()]
    expected={(t['name'],mode) for t in spec['targets'] for mode in MODES
              if t['kind']=='linear' or mode=='w4a16'}
    actual=[(r['target']['name'],r['mode']) for r in rows]
    assert len(actual)==len(set(actual)) and set(actual)==expected
    ids={s['id'] for s in spec['samples']}
    fold_by_id={s['id']:s['fold'] for s in spec['samples']}
    assert len(ids)==len(spec['samples'])
    for row in rows:
        assert {e['id'] for e in row['examples']}==ids and len(row['examples'])==len(ids)
        for example in row['examples']:
            assert example['fold']==fold_by_id[example['id']]
            assert np.isfinite([example[k] for k in ('kl','nll','nll_delta','argmax_agreement')]).all()
        for fold,metrics in row['metrics'].items():
            selected=[e for e in row['examples'] if fold=='all' or e['fold']==fold]
            assert metrics['examples']==len(selected)
            assert metrics['kl']==statistics.mean(e['kl'] for e in selected)
            assert metrics['nll_delta']==statistics.mean(e['nll_delta'] for e in selected)
    assert summary['model_parameters_restored'] and summary['fp16_control_repeated_exactly']
    assert summary['finetuning_steps']==0
    projections=[]
    for mode in MODES:
        labels=sorted({r['target']['projection'] for r in rows if r['mode']==mode
                       and r['target']['level']=='leaf' and r['target']['block'] is not None})
        for label in labels:
            selected=[r for r in rows if r['mode']==mode and r['target']['level']=='leaf'
                      and r['target']['block'] is not None and r['target']['projection']==label]
            top=max(selected,key=lambda r:r['metrics']['all']['kl'])
            projections.append({'mode':mode,'projection':label,'blocks':len(selected),
                'median_kl':statistics.median(r['metrics']['all']['kl'] for r in selected),
                'mean_kl':statistics.mean(r['metrics']['all']['kl'] for r in selected),
                'max_kl':top['metrics']['all']['kl'],'most_sensitive_block':top['target']['block']})
    comparison=paired_groups(rows,spec['samples'])
    verification={'time':now(),'status':'complete','finetuning_steps':0,'cases':len(rows),
                  'examples':len(ids),'fold_counts':dict(Counter(fold_by_id.values())),
                  'all_example_ids_folds_and_metric_aggregates_verified':True,
                  'source_hashes_match':True,'projection_statistics':projections,
                  'paired_group_bootstrap':comparison,
                  'reporter_sha256':sha256(Path(__file__))}
    write_json(directory/'verification.json',verification)
    write_report(directory,rows,summary)
    baseline=json.loads((directory/'baseline.json').read_text())
    control=baseline['public_forward_control']
    lines=['','## 실행 경로 검증','',
           '오른쪽 padding을 사용하고 유효한 정답 위치만 평가하므로, 미래의 padding을 명시적으로 가리는 mask를 생략해 causal SDPA 경로를 사용했다. 모든 개입과 FP16 기준에 같은 경로를 적용했다.',
           f"공개 masked forward와 비교한 첫 {control['checked_examples']}개 예제의 평균 KL 차이는 {control['public_forward_mean_kl']:.6g}였다. 어휘 head에 입력하는 행 수도 달라질 수 있어 공개 forward와의 bit 일치를 주장하지 않는다.",
           f"재사용한 FP16 prefix와 head 입력은 첫 배치에서 재계산 결과와 bit 단위로 일치했다. 전체 {summary['examples']}개 예제의 개입 전후 FP16 출력은 정확히 일치했고 파라미터 hash도 복구됐다.",
           '이 분석은 FP16 나머지 모델에서 단일 대상을 바꾼 민감도이며, 전체 INT4 모델에서 한 레이어를 복구한 성능 변화와는 구분한다.',
           '','## Projection별 집계','','22개 블록의 개별 레이어를 집계했다. 평균과 중앙값 차이가 크면 일부 블록의 큰 영향이 평균을 끌어올린 경우다.','',
           '| 모드 | Projection | KL 중앙값 | KL 평균 | 최대 KL 블록 |',
           '| --- | --- | ---: | ---: | ---: |']
    for row in projections:
        lines.append(f"| {row['mode'].upper()} | {row['projection']} | {row['median_kl']:.6g} | {row['mean_kl']:.6g} | {row['most_sensitive_block']} |")
    lines += ['','## Attention·FFN의 예제별 대응 비교','',
              '같은 예제에서 22개 블록의 Attention 묶음 KL 평균과 FFN 묶음 KL 평균을 비교했다.',
              'FFN−Attention 값이 양수면 이번 개입에서 FFN의 출력 변화가 더 컸다는 뜻이다. 파라미터 수·저장 비용을 맞춘 비교는 아니다.',
              '95% 구간은 예제를 5,000번 복원 추출한 paired bootstrap이다. 고정된 모델·calibration 분포에 대한 변동이며 전체 과제 정확도의 신뢰구간이 아니다.','',
              '| 모드 | 집합 | Attention 평균 KL | FFN 평균 KL | FFN−Attention | 95% 구간 |',
              '| --- | --- | ---: | ---: | ---: | --- |']
    for mode,folds in comparison.items():
        for fold,result in folds.items():
            lo,hi=result['paired_bootstrap_95_ci']
            lines.append(f"| {mode.upper()} | {fold} | {result['mean_attention_kl']:.6g} | {result['mean_ffn_kl']:.6g} | "
                         f"{result['mean_ffn_minus_attention_kl']:+.6g} | [{lo:+.6g}, {hi:+.6g}] |")
    with (directory/'report.md').open('a') as output:
        output.write('\n'.join(lines)+'\n')
    print(f'Verified {len(rows)} cases × {len(ids)} examples: {directory / "report.md"}')


if __name__=='__main__':
    main()
