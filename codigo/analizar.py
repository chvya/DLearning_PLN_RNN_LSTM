"""Produce tablas y figuras desde resultados reales, sin volver a entrenar."""
import csv
import json
from pathlib import Path
import statistics as st
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pln import ROOT, OUT, predict, save_json


def main():
    rows = json.loads((OUT/'evaluacion_test.json').read_text(encoding='utf-8'))
    figures = OUT/'figuras'
    figures.mkdir(parents=True, exist_ok=True)
    summary = {}
    for kind in ('rnn','lstm'):
        subset = [r for r in rows if r['modelo']==kind and r['contexto']==12]
        summary[kind] = {metric: {'media': st.mean(r['test'][metric] for r in subset),
            'desviacion_muestral': st.stdev(r['test'][metric] for r in subset)}
            for metric in ('nll','perplejidad','top1','top5')}
        summary[kind]['parametros'] = subset[0]['parametros']
        summary[kind]['segundos_media'] = st.mean(r['segundos_entrenamiento'] for r in subset)
    save_json(OUT/'resumen_comparacion.json',summary)
    with (OUT/'metricas.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['modelo','contexto','semilla','NLL','PPL','top1_pct','top5_pct','parametros','entrenamiento_s','latencia_mediana_ms'])
        for r in rows:
            t=r['test']
            writer.writerow([r['nombre'],r.get('contexto',1),r.get('semilla',''),t['nll'],t['perplejidad'],100*t['top1'],100*t['top5'],r.get('parametros',''),r.get('segundos_entrenamiento',''),r.get('latencia_mediana_ms','')])
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#576b7d','#007f86','#d78335']
    fig, axes = plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    names=['Bigrama','RNN','LSTM']
    for ax,metric,label,scale in zip(axes,['perplejidad','top5'],['Perplejidad (menor es mejor)','Top-5 léxico (%)'],[1,100]):
        values=[rows[0]['test'][metric]*scale]+[summary[k][metric]['media']*scale for k in ('rnn','lstm')]
        errors=[0]+[summary[k][metric]['desviacion_muestral']*scale for k in ('rnn','lstm')]
        bars=ax.bar(names,values,color=colors,yerr=errors,capsize=4)
        ax.bar_label(bars,fmt='%.2f',padding=5)
        ax.set_ylabel(label); ax.set_ylim(0,max(values)*1.2)
    fig.savefig(figures/'comparacion.pdf'); fig.savefig(figures/'comparacion.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained')
    for kind,color in zip(('rnn','lstm'),colors[1:]):
        for seed in (17,29,43):
            history=json.loads((OUT/f'{kind}_c12_s{seed}_historial.json').read_text())
            ax.plot([h['epoca'] for h in history],[h['val']['nll'] for h in history],marker='o',color=color,alpha=.75,label=f'{kind.upper()} {seed}')
    ax.set_xlabel('Época'); ax.set_ylabel('NLL de validación'); ax.set_xticks(range(1,6)); ax.legend(ncol=2,fontsize=8)
    fig.savefig(figures/'aprendizaje.pdf'); fig.savefig(figures/'aprendizaje.png',dpi=180); plt.close(fig)
    examples=[predict(t) for t in ['', 'el presidente del gobierno', 'la universidad de', 'los estudiantes de software', 'Riobamba ESPOCH Chimborazo']]
    save_json(OUT/'ejemplos_demo.json',examples)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
