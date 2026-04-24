import pandas as pd, numpy as np
bloc = ['BAN','BHU','IND','MLD','NEP','PAK','SRI','BRU','CAM','INO','LAO','MAL','PHI','SIN','VIE','THA']
names = {'BAN':'Bangladesh','BHU':'Bhutan','IND':'India','MLD':'Maldives','NEP':'Nepal',
         'PAK':'Pakistan','SRI':'Sri Lanka','BRU':'Brunei','CAM':'Cambodia','INO':'Indonesia',
         'LAO':'Lao PDR','MAL':'Malaysia','PHI':'Philippines','SIN':'Singapore',
         'VIE':'Vietnam','THA':'Thailand'}

df = pd.read_csv('outputs/gvc_baseline.csv')
cf = pd.read_csv('outputs/counterfactual_2021.csv')

print("=== Bloc Average GVC Metrics by Year (%) ===")
sub = df[df['economy'].isin(bloc)].groupby('year')[['gvc_total','gvc_back','gvc_fwd','gross_exp']].mean()
sub[['gvc_total','gvc_back','gvc_fwd']] *= 100
print(sub.round(3).to_string())

print("\n=== 2021 Country GVC Participation (%) ===")
s21 = df[(df['year']==2021) & (df['economy'].isin(bloc))][['economy','gvc_total','gvc_back','gvc_fwd','gross_exp']].copy()
s21['gvc_total_pct'] = s21['gvc_total']*100
s21['gvc_back_pct']  = s21['gvc_back']*100
s21['gvc_fwd_pct']   = s21['gvc_fwd']*100
print(s21[['economy','gvc_total_pct','gvc_back_pct','gvc_fwd_pct','gross_exp']].sort_values('gvc_total_pct',ascending=False).to_string(index=False))

print("\n=== Counterfactual Delta (ppt) ===")
act21 = df[df['year']==2021].set_index('economy')
cf21  = cf.set_index('economy')
for c in bloc:
    if c in act21.index and c in cf21.index:
        d_tot  = (cf21.loc[c,'gvc_total'] - act21.loc[c,'gvc_total'])*100
        d_back = (cf21.loc[c,'gvc_back']  - act21.loc[c,'gvc_back'])*100
        d_fwd  = (cf21.loc[c,'gvc_fwd']   - act21.loc[c,'gvc_fwd'])*100
        print(f"{names.get(c,c):12s} ({c})  total={d_tot:+.4f}  back={d_back:+.4f}  fwd={d_fwd:+.4f}")

bau  = pd.read_csv('outputs/forecast_bau.csv')
intg = pd.read_csv('outputs/forecast_integrated.csv')
print("\n=== Forecast 2022-2024: Bloc Average GVC (%) ===")
bb = bau[bau['economy'].isin(bloc)].groupby('year')[['gvc_total','gvc_back','gvc_fwd']].mean()*100
ii = intg[intg['economy'].isin(bloc)].groupby('year')[['gvc_total','gvc_back','gvc_fwd']].mean()*100
print("BAU:\n", bb.round(3).to_string())
print("Integrated:\n", ii.round(3).to_string())
print("Delta (Integrated - BAU):\n", (ii-bb).round(4).to_string())
