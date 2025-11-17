import streamlit as st
import pandas as pd
import requests
import re
from lxml import etree
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import base64
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fund Manager 13F Dashboard", layout="wide")

with open('assets/style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def qtr_to_end_date(qtr_str: str) -> str:
    m = re.match(r"(\d{4})Q([1-4])", qtr_str)
    if not m:
        raise ValueError('Quarter must be like 2025Q3')
    y = int(m.group(1))
    q = int(m.group(2))
    month = q * 3
    # last day by month
    if month in (4,6,9,11):
        day = 30
    elif month == 2:
        day = 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28
    else:
        day = 31
    return f"{y}-{month:02d}-{day:02d}"

def prev_qtr(qtr_str: str, n: int = 1) -> str:
    y = int(qtr_str[:4])
    q = int(qtr_str[5])
    for _ in range(n):
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return f"{y}Q{q}"

@st.cache_data(ttl=60*60*6)
def search_cik_by_name(name):
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {"action":"getcompany","company":name}
    headers = {"User-Agent":"fin-dashboard/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    text = r.text
    m = re.search(r"CIK=0*(\d+)", text)
    if m:
        return m.group(1)
    m2 = re.search(r"CIK=(\d+)", text)
    return m2.group(1) if m2 else None

@st.cache_data(ttl=60*60*6)
def get_13f_filings(cik):
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {"action":"getcompany","CIK":cik,"type":"13F-HR","owner":"exclude","count":200}
    headers = {"User-Agent":"fin-dashboard/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    links = re.findall(r'href="(/Archives/edgar/data/\d+/\d+/[^\"]+)"', r.text)
    full_links = ["https://www.sec.gov" + l for l in links]
    return full_links

@st.cache_data(ttl=60*60*6)
def download_and_parse_13f_xml(filing_url):
    headers = {"User-Agent":"fin-dashboard/1.0"}
    r = requests.get(filing_url, headers=headers, timeout=30)
    # find xml or information table
    xml_links = re.findall(r'href="(.*?\.xml)"', r.text)
    xml_link = None
    if xml_links:
        xml_link = 'https://www.sec.gov' + xml_links[0]
    else:
        txt_links = re.findall(r'href="(.*?\.txt)"', r.text)
        xml_candidates = [l for l in txt_links if 'information' in l.lower() or 'infotable' in l.lower()]
        if xml_candidates:
            xml_link = 'https://www.sec.gov' + xml_candidates[0]
    if not xml_link:
        return None, None
    r2 = requests.get(xml_link, headers=headers, timeout=30)
    try:
        tree = etree.fromstring(r2.content)
    except Exception:
        return None, None
    period = tree.findtext('.//{*}periodOfReport')
    records = []
    for it in tree.findall('.//{*}infotable'):
        name = it.findtext('.//{*}nameOfIssuer')
        cusip = it.findtext('.//{*}cusip')
        value = it.findtext('.//{*}value')
        ssh = it.findtext('.//{*}shrsOrPrnAmt/{*}sshPrnamt')
        putcall = it.findtext('.//{*}putCall')
        title = it.findtext('.//{*}titleOfClass')
        records.append({
            "name": name,
            "cusip": cusip,
            "value_k": float(value) if value and value.isdigit() else (float(value) if value and re.match(r"^[0-9,.]+$", value) else np.nan),
            "shares": float(ssh) if ssh and re.match(r"^[0-9,.]+$", str(ssh)) else np.nan,
            "putcall": putcall,
            "title": title,
        })
    df = pd.DataFrame(records)
    if df.empty:
        return None, period
    df['value_usd'] = df['value_k'] * 1000
    total = df['value_usd'].sum()
    df['weight'] = df['value_usd'] / total if total > 0 else 0
    return df, period

def fetch_manager_holdings_multi(manager_name, n_quarters=4):
    cik = search_cik_by_name(manager_name)
    if not cik:
        return None, f"CIK not found for {manager_name}"
    filings = get_13f_filings(cik)
    qdfs = []
    for f in filings:
        try:
            df, period = download_and_parse_13f_xml(f)
            if df is not None and period:
                # normalize period to YYYY-MM-DD if possible
                p = None
                if re.match(r"\d{4}-\d{2}-\d{2}", period):
                    p = period
                else:
                    # try parse other formats
                    try:
                        p = pd.to_datetime(period).strftime('%Y-%m-%d')
                    except Exception:
                        p = None
                if p:
                    qdfs.append((p, df))
            if len(qdfs) >= n_quarters * 2:
                # gather more than needed then sort
                break
        except Exception:
            continue
    if not qdfs:
        return None, "No parsable 13F XML found"
    qdfs = sorted(qdfs, key=lambda x: x[0], reverse=True)
    # take unique periods
    seen = set()
    out = []
    for p, d in qdfs:
        if p in seen:
            continue
        seen.add(p)
        out.append((p, d))
        if len(out) >= n_quarters:
            break
    return out, None

def compute_qoq_changes(latest_df, prev_df):
    # merge on cusip where available, fallback to name
    key = 'cusip' if latest_df['cusip'].notna().any() and prev_df['cusip'].notna().any() else 'name'
    l = latest_df.copy()
    p = prev_df.copy()
    l = l.set_index(key)
    p = p.set_index(key)
    combined = l[['value_usd','shares','weight']].join(p[['value_usd','shares','weight']], lsuffix='_t', rsuffix='_p', how='outer')
    combined = combined.fillna(0)
    combined['abs_change_usd'] = combined['value_usd_t'] - combined['value_usd_p']
    combined['pct_change'] = np.where(combined['value_usd_p']>0, combined['abs_change_usd']/combined['value_usd_p'], np.nan)
    combined['new'] = (combined['value_usd_p']==0) & (combined['value_usd_t']>0)
    combined['exited'] = (combined['value_usd_t']==0) & (combined['value_usd_p']>0)
    combined['agg_add_30'] = combined['pct_change'] > 0.30
    combined['agg_add_50'] = combined['pct_change'] > 0.50
    combined['agg_red_30'] = combined['pct_change'] < -0.30
    combined['agg_red_50'] = combined['pct_change'] < -0.50
    combined['rank_t'] = combined['value_usd_t'].rank(ascending=False, method='min')
    combined['rank_p'] = combined['value_usd_p'].rank(ascending=False, method='min')
    return combined.reset_index()

st.title('Fund Manager 13F Dashboard — Dark Mode')

with st.sidebar:
    st.header('Inputs')
    managers_input = st.text_area('Managers (one per line)', value='Berkshire Hathaway\nPershing Square')
    target_quarter = st.text_input('Target Quarter', value='2025Q3')
    min_conviction_weight = st.number_input('Min conviction weight (decimal)', value=0.02, step=0.01)
    st.markdown('---')
    st.subheader('Liquidity Controls')
    sell_rate_options = [5,10,25,50]
    preset = st.selectbox('Sell-rate preset', options=['Conservative','Standard','Aggressive'], index=1)
    preset_map = {'Conservative':[5,10], 'Standard':[10,25], 'Aggressive':[25,50]}
    sell_rates = st.multiselect('Sell rates (% of ADV per day)', options=sell_rate_options, default=preset_map.get(preset, [10,25]))
    adv_lookback = st.selectbox('ADV lookback (days)', options=[10,20,60], index=1)
    single_manager_threshold = st.number_input('Single manager float threshold (%)', value=10.0, step=1.0)
    aggregate_float_threshold = st.number_input('Aggregate float threshold (%)', value=50.0, step=1.0)
    run = st.button('Run Analysis')

if run:
    managers = [m.strip() for m in managers_input.splitlines() if m.strip()]
    st.markdown(f"**Target quarter:** {target_quarter}  •  **Min conviction weight:** {min_conviction_weight*100:.1f}%")
    holdings_by_manager = {}
    errors = []
    multi_q_by_manager = {}
    for mgr in managers:
        with st.spinner(f'Fetching {mgr}'):
            out, err = fetch_manager_holdings_multi(mgr, n_quarters=4)
            if err:
                errors.append((mgr, err))
            else:
                multi_q_by_manager[mgr] = out
                # set latest as target period (first)
                latest_period, latest_df = out[0]
                holdings_by_manager[mgr] = {'period': latest_period, 'df': latest_df}
    if errors:
        for e in errors:
            st.error(f"{e[0]}: {e[1]}")

    # Prepare consensus breadth using latest period across managers
    all_latest = []
    for mgr, info in holdings_by_manager.items():
        tmp = info['df'].copy()
        tmp['manager'] = mgr
        all_latest.append(tmp[['name','cusip','value_usd','weight','manager']])
    consensus_df = pd.concat(all_latest, ignore_index=True) if all_latest else pd.DataFrame()
    breadth = consensus_df.groupby('name').manager.nunique().reset_index().rename(columns={'manager':'breadth'})

    # Compute previous-quarter breadth for momentum across managers (one quarter back)
    prev_all = []
    for mgr, out in multi_q_by_manager.items():
        if len(out) >= 2:
            prev_period, prev_df = out[1]
            tmp = prev_df.copy(); tmp['manager']=mgr
            prev_all.append(tmp[['name','manager']])
    prev_consensus_df = pd.concat(prev_all, ignore_index=True) if prev_all else pd.DataFrame()
    prev_breadth = prev_consensus_df.groupby('name').manager.nunique().reset_index().rename(columns={'manager':'breadth_p'})
    breadth = breadth.merge(prev_breadth, on='name', how='left').fillna(0)
    breadth['breadth_momentum'] = breadth['breadth'] - breadth['breadth_p']

    # Generate signals per manager and global signals
    signals = {
        'high_conviction_new_buys': [],
        'agg_add_30': [],
        'agg_add_50': [],
        'agg_red_30': [],
        'agg_red_50': [],
        'full_liquidations': [],
        'new_guard': []
    }

    # Build a lookup of breadth for use in new guard
    breadth_lookup = breadth.set_index('name')['breadth'].to_dict()

    def compute_crowding(consensus_df, multi_q_by_manager):
        # consensus_df: latest combined holdings rows with manager
        # returns DataFrame with crowding metrics per name
        if consensus_df.empty:
            return pd.DataFrame()
        # total value across managers per name
        total_by_name = consensus_df.groupby('name').value_usd.sum().rename('total_value')
        # HHI across managers for each name: sum over managers of (value_i / total)^2
        hhi_rows = []
        for name, group in consensus_df.groupby('name'):
            vals = group.groupby('manager').value_usd.sum()
            total = vals.sum()
            if total <= 0:
                hhi = 0.0
            else:
                shares = vals / total
                hhi = float((shares ** 2).sum())
            hhi_rows.append((name, hhi, total))
        hhi_df = pd.DataFrame(hhi_rows, columns=['name','hhi','total_value'])

        # previous quarter total_value (for momentum in concentration)
        prev_rows = []
        for mgr, out in multi_q_by_manager.items():
            if len(out) >= 2:
                prev_period, prev_df = out[1]
                tmp = prev_df.copy(); tmp['manager']=mgr
                prev_rows.append(tmp[['name','value_usd']])
        prev_all = pd.concat(prev_rows, ignore_index=True) if prev_rows else pd.DataFrame()
        prev_tot = prev_all.groupby('name').value_usd.sum().rename('total_prev') if not prev_all.empty else pd.Series(dtype=float)

        res = hhi_df.set_index('name')
        res['breadth'] = breadth.set_index('name')['breadth']
        res['breadth_momentum'] = breadth.set_index('name')['breadth_momentum']
        res['total_prev'] = prev_tot
        res['total_prev'] = res['total_prev'].fillna(0)
        res['total_change'] = res['total_value'] - res['total_prev']
        res['total_pct_change'] = np.where(res['total_prev']>0, res['total_change']/res['total_prev'], np.nan)

        # normalize components to [0,1]
        max_breadth = max(1, res['breadth'].max())
        res['norm_breadth'] = res['breadth'] / max_breadth
        # HHI ranges 0..1; use directly
        res['norm_hhi'] = res['hhi']
        # normalize total_value
        max_total = max(1.0, res['total_value'].max())
        res['norm_total'] = res['total_value'] / max_total

        # crowding score composition: 50% breadth, 30% HHI, 20% total momentum (positive)
        # breadth increases crowding; HHI increases crowding; rising total_value increases crowding
        res['pos_mom'] = res['total_change'].clip(lower=0) / (res['total_value'] + 1e-9)
        res['norm_pos_mom'] = res['pos_mom'] / max(1e-9, res['pos_mom'].max())
        res['crowding_score'] = 0.5 * res['norm_breadth'] + 0.3 * res['norm_hhi'] + 0.2 * res['norm_pos_mom']

        # flags
        res['is_most_crowded'] = res['crowding_score'] >= res['crowding_score'].quantile(0.9)
        res['is_rising'] = (res['breadth_momentum'] > 0) & (res['total_change'] > 0)
        res['is_uncrowded_accumulating'] = (res['breadth'] < 3) & (res['total_change'] > 0)
        return res.reset_index()

    crowd_df = compute_crowding(consensus_df, multi_q_by_manager)

    for mgr, out in multi_q_by_manager.items():
        latest_period, latest_df = out[0]
        prev_df = out[1][1] if len(out) >= 2 else pd.DataFrame(columns=latest_df.columns)
        combined = compute_qoq_changes(latest_df, prev_df)
        combined['manager'] = mgr
        # high conviction new buys
        hc_new = combined[(combined['new']) & (combined['weight_t'] >= min_conviction_weight)]
        for idx, r in hc_new.iterrows():
            signals['high_conviction_new_buys'].append({'manager':mgr,'name':r[combined.columns[0]],'weight':r['weight_t'],'value_usd':r['value_usd_t']})
        # aggressive adds/reductions
        for idx, r in combined.iterrows():
            name = r[combined.columns[0]]
            if r['agg_add_30']:
                signals['agg_add_30'].append({'manager':mgr,'name':name,'pct_change':r['pct_change'],'weight':r['weight_t']})
            if r['agg_add_50']:
                signals['agg_add_50'].append({'manager':mgr,'name':name,'pct_change':r['pct_change'],'weight':r['weight_t']})
            if r['agg_red_30']:
                signals['agg_red_30'].append({'manager':mgr,'name':name,'pct_change':r['pct_change'],'weight':r['weight_t']})
            if r['agg_red_50']:
                signals['agg_red_50'].append({'manager':mgr,'name':name,'pct_change':r['pct_change'],'weight':r['weight_t']})
            if r['exited']:
                signals['full_liquidations'].append({'manager':mgr,'name':name,'value_prev':r['value_usd_p']})
        # new guard candidates (new & low breadth across managers)
        new_candidates = combined[combined['new']].reset_index()
        for idx, r in new_candidates.iterrows():
            name = r[combined.columns[0]]
            br = breadth_lookup.get(name, 0)
            if br < 3:
                signals['new_guard'].append({'manager':mgr,'name':name,'breadth':br,'weight':r['weight_t']})

    # UI tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Manager Snapshots","Consensus & Crowding","New Guard & Fading Consensus","Liquidity & Ownership","Filings Timeline"])

    with tab1:
        st.header('Manager Snapshots')
        for mgr, out in multi_q_by_manager.items():
            latest_period, latest_df = out[0]
            st.subheader(f"{mgr} — {latest_period}")
            col1, col2 = st.columns([2,1])
            with col1:
                st.markdown('<div class="sb-card">', unsafe_allow_html=True)
                top10 = latest_df.sort_values('value_usd', ascending=False).head(10)
                st.write(top10[['name','cusip','value_usd','weight']].rename(columns={'value_usd':'value ($)','weight':'weight (%)'}).to_html(index=False), unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.metric('Portfolio Value (reported)', f"${latest_df['value_usd'].sum():,.0f}")
                st.metric('Positions', len(latest_df))

    with tab2:
        st.header('Consensus & Crowding')
        st.markdown('**Holder Breadth (top names)**')
        st.dataframe(breadth.sort_values('breadth', ascending=False).head(50))
        st.markdown('**Crowding Summary**')
        if not crowd_df.empty:
            top_crowded = crowd_df.sort_values('crowding_score', ascending=False).head(20)
            rising = crowd_df[crowd_df['is_rising']].sort_values('crowding_score', ascending=False).head(20)
            uncrowded_acc = crowd_df[crowd_df['is_uncrowded_accumulating']].sort_values('total_change', ascending=False).head(20)
            st.markdown('**Most Crowded Names (top 20)**')
            st.table(top_crowded[['name','breadth','hhi','total_value','crowding_score']])
            st.markdown('*Insight:* The table highlights names with the highest composite crowding score (breadth, HHI, and recent accumulation).')
            st.markdown('**Rising Crowding Names**')
            st.table(rising[['name','breadth','breadth_momentum','total_change','crowding_score']])
            st.markdown('*Insight:* Names with increasing holder breadth and rising aggregate position value across managers.')
            st.markdown('**Uncrowded But Accumulating**')
            st.table(uncrowded_acc[['name','breadth','total_change']])
            st.markdown('*Insight:* Smaller breadth (<3) but with positive net accumulation across the managers in our set — watchlist for potential New Guard leaders.')
        else:
            st.write('*Crowding metrics unavailable (insufficient data).*')

    with tab3:
        st.header('New Guard & Fading Consensus')
        st.markdown('**High Conviction New Buys**')
        if signals['high_conviction_new_buys']:
            df_hc = pd.DataFrame(signals['high_conviction_new_buys'])
            st.table(df_hc)
            csv_hc = df_hc.to_csv(index=False).encode('utf-8')
            st.download_button('Download High-Conviction New Buys CSV', data=csv_hc, file_name='high_conviction_new_buys.csv', mime='text/csv')
        else:
            st.write('*No high conviction new buys found.*')
        st.markdown('**Aggressive Adds (>30% / >50%)**')
        df_add30 = pd.DataFrame(signals['agg_add_30'])
        st.table(df_add30.head(50))
        if not df_add30.empty:
            st.download_button('Download Aggressive Adds (>30%) CSV', data=df_add30.to_csv(index=False).encode('utf-8'), file_name='agg_add_30.csv', mime='text/csv')
        st.markdown('**Aggressive Reductions (>30% / >50%)**')
        df_red30 = pd.DataFrame(signals['agg_red_30'])
        st.table(df_red30.head(50))
        if not df_red30.empty:
            st.download_button('Download Aggressive Reductions (>30%) CSV', data=df_red30.to_csv(index=False).encode('utf-8'), file_name='agg_red_30.csv', mime='text/csv')
        st.markdown('**Full Liquidations**')
        df_liq = pd.DataFrame(signals['full_liquidations'])
        st.table(df_liq.head(50))
        if not df_liq.empty:
            st.download_button('Download Full Liquidations CSV', data=df_liq.to_csv(index=False).encode('utf-8'), file_name='full_liquidations.csv', mime='text/csv')
        st.markdown('**New Guard (Uncrowded Accumulators)**')
        df_newguard = pd.DataFrame(signals['new_guard'])
        st.table(df_newguard.head(50))
        if not df_newguard.empty:
            st.download_button('Download New Guard CSV', data=df_newguard.to_csv(index=False).encode('utf-8'), file_name='new_guard.csv', mime='text/csv')
        # combined signals download
        all_signals = []
        for sname, svals in signals.items():
            for item in svals:
                row = item.copy()
                row['signal_type'] = sname
                all_signals.append(row)
        if all_signals:
            df_all_signals = pd.DataFrame(all_signals)
            st.download_button('Download All Signals CSV', data=df_all_signals.to_csv(index=False).encode('utf-8'), file_name='all_signals.csv', mime='text/csv')

    with tab4:
        st.header('Liquidity & Ownership')
        st.write('Float ownership calculations use `yfinance` where available. This may be incomplete.')
        # Build liquidity table for each consensus name
        @st.cache_data(ttl=60*60*6)
        def search_yahoo_ticker(query):
            # best-effort Yahoo search endpoint
            url = f"https://query2.finance.yahoo.com/v1/finance/search?q={requests.utils.quote(query)}"
            headers = {"User-Agent":"fin-dashboard/1.0"}
            try:
                r = requests.get(url, headers=headers, timeout=10)
                data = r.json()
                quotes = data.get('quotes', [])
                if not quotes:
                    return None
                # pick best match by exact company name or first symbol
                for q in quotes:
                    if 'shortname' in q and q['shortname'] and query.lower() in q['shortname'].lower():
                        return q.get('symbol')
                return quotes[0].get('symbol')
            except Exception:
                return None

        
        def map_cusip_to_ticker_openfigi(cusip: str):
            api_key = os.environ.get('OPENFIGI_APIKEY')
            if not api_key or not cusip:
                return None
            url = 'https://api.openfigi.com/v3/mapping'
            headers = {'Content-Type':'application/json', 'X-OPENFIGI-APIKEY': api_key}
            body = [{"idType":"ID_CUSIP","idValue":cusip}]
            try:
                r = requests.post(url, json=body, headers=headers, timeout=10)
                data = r.json()
                if data and isinstance(data, list) and 'data' in data[0] and data[0]['data']:
                    entry = data[0]['data'][0]
                    # openfigi returns 'ticker' or 'exchCode' fields
                    return entry.get('ticker') or entry.get('exchCode')
            except Exception:
                return None
            return None

        @st.cache_data(ttl=60*60*6)
        def fetch_liquidity_for_ticker(ticker, adv_window=20):
            try:
                tk = yf.Ticker(ticker)
                info = tk.get_info() if hasattr(tk, 'get_info') else tk.info
            except Exception:
                info = {}
            # try to get floatShares or sharesOutstanding
            float_shares = info.get('floatShares') or info.get('float') or info.get('sharesFloat')
            shares_out = info.get('sharesOutstanding')
            # fetch recent volume over adv_window days
            adv = None
            try:
                hist = tk.history(period=f"{max(30, adv_window)}d", interval='1d')
                if not hist.empty:
                    adv = float(hist['Volume'].dropna().tail(adv_window).mean())
            except Exception:
                adv = None
            return {'float_shares': float_shares, 'shares_out': shares_out, 'adv': adv}

        # assemble liquidity info for consensus names
        liquidity_rows = []
        if not consensus_df.empty:
            # compute aggregate shares per name across managers
            agg_shares = consensus_df.groupby('name').shares.sum().rename('agg_shares')
            # per-manager shares too
            per_mgr = consensus_df.groupby(['name','manager']).shares.sum().rename('mgr_shares').reset_index()

            # build quarter-series totals for sparklines
            def build_name_quarter_series(name):
                period_totals = {}
                for mgr, out in multi_q_by_manager.items():
                    for p, df in out:
                        val = df[df['name']==name]['value_usd'].sum() if not df.empty else 0
                        period_totals[p] = period_totals.get(p, 0) + val
                # sort by period chronologically
                periods_sorted = sorted(period_totals.keys())
                return [period_totals[p] for p in periods_sorted]

            def sparkline_svg(values, width=120, height=24, color='#2EE4D4'):
                if not values:
                    return ''
                plt.ioff()
                fig, ax = plt.subplots(figsize=(width/72, height/72), dpi=72)
                ax.plot(values, color=color, linewidth=1.2)
                ax.fill_between(range(len(values)), values, color=color, alpha=0.05)
                ax.axis('off')
                buf = io.BytesIO()
                fig.savefig(buf, format='svg', bbox_inches='tight', transparent=True)
                plt.close(fig)
                svg = buf.getvalue().decode('utf-8')
                b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
                return f"<img src='data:image/svg+xml;base64,{b64}' style='height:{height}px'/>"

            for name, row in agg_shares.reset_index().iterrows():
                nm = row['name']
                agg = row['agg_shares']
                # attempt CUSIP->ticker via OpenFIGI first
                cusip_vals = consensus_df[consensus_df['name']==nm]['cusip'].unique()
                cusip = cusip_vals[0] if len(cusip_vals)>0 else None
                ticker = None
                if cusip:
                    ticker = map_cusip_to_ticker_openfigi(cusip)
                if not ticker:
                    ticker = search_yahoo_ticker(nm)
                liq = fetch_liquidity_for_ticker(ticker, adv_window=adv_lookback) if ticker else {'float_shares':None,'shares_out':None,'adv':None}
                float_shares = liq.get('float_shares') or liq.get('shares_out')
                adv = liq.get('adv')
                pct_of_float = (agg / float_shares) if float_shares and float_shares>0 else None
                # per-manager flags
                mgrs = per_mgr[per_mgr['name']==nm][['manager','mgr_shares']].to_dict('records') if not per_mgr.empty else []
                any_single_flag = any([(m['mgr_shares']/float_shares if float_shares and float_shares>0 else 0) > (single_manager_threshold/100.0) for m in mgrs])
                aggregate_flag = (pct_of_float is not None and pct_of_float > (aggregate_float_threshold/100.0))
                # days to exit for selected sell rates
                dte_map = {}
                if adv and adv>0 and sell_rates:
                    for rate in sell_rates:
                        try:
                            rfrac = float(rate)/100.0
                            dte = agg / (adv * rfrac) if rfrac>0 else None
                        except Exception:
                            dte = None
                        dte_map[f'days_to_exit_{rate}pctADV'] = dte
                # trend sparkline
                series = build_name_quarter_series(nm)
                spark = sparkline_svg(series)
                row = {
                    'name': nm,
                    'ticker': ticker,
                    'agg_shares': agg,
                    'float_shares': float_shares,
                    'pct_of_float': pct_of_float,
                    'any_single_flag': any_single_flag,
                    'aggregate_flag': aggregate_flag,
                    'adv': adv,
                    'trend': spark,
                }
                row.update(dte_map)
                liquidity_rows.append(row)
        liq_df = pd.DataFrame(liquidity_rows)
        if not liq_df.empty:
            # format and show risk flags
            liq_df_display = liq_df.copy()
            # add colored HTML cells for conditional coloring
            def color_pct_of_float(x):
                if pd.isna(x):
                    return 'N/A'
                try:
                    v = float(x)
                except Exception:
                    return 'N/A'
                if v > (aggregate_float_threshold/100.0):
                    color = '#FF4D4D'  # red
                elif v > (single_manager_threshold/100.0):
                    color = '#FFC940'  # yellow
                else:
                    color = '#00C676'  # green
                return f"<span style='color:{color};font-weight:600'>{v:.2%}</span>"

            def color_adv(x):
                if pd.isna(x):
                    return 'N/A'
                return f"<span style='color:#A8A8A8'>{int(x):,}</span>"

            liq_df_display['pct_of_float'] = liq_df_display['pct_of_float'].apply(lambda x: color_pct_of_float(x) if pd.notna(x) else 'N/A')
            liq_df_display['adv'] = liq_df_display['adv'].apply(lambda x: color_adv(x) if pd.notna(x) else 'N/A')
            # format dynamic dte columns
            for rate in sell_rates:
                col = f'days_to_exit_{rate}pctADV'
                if col in liq_df_display.columns:
                    liq_df_display[col] = liq_df_display[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
            # produce risk badge and keep trend HTML unescaped
            def badge_html(row):
                if row.get('aggregate_flag'):
                    return f"<span style='background:#3A1B1B;color:#FF4D4D;padding:4px 8px;border-radius:10px;font-weight:700'>AGG&gt;{int(aggregate_float_threshold)}%</span>"
                if row.get('any_single_flag'):
                    return f"<span style='background:#2A2210;color:#FFC940;padding:4px 8px;border-radius:10px;font-weight:700'>SGL&gt;{int(single_manager_threshold)}%</span>"
                return f"<span style='background:#072214;color:#00C676;padding:4px 8px;border-radius:10px;font-weight:700'>OK</span>"

            liq_df_display['risk'] = [badge_html(r) for r in liquidity_rows]
            # move trend to displayable HTML (it's already HTML img)
            # ensure numeric DTE columns formatted
            display_df = liq_df_display.sort_values('pct_of_float', ascending=False).head(50).copy()
            # reorder columns for presentation
            cols = [c for c in ['risk','name','ticker','trend','adv','pct_of_float','agg_shares','float_shares'] if c in display_df.columns]
            # add dynamic DTE columns
            dte_cols = [c for c in display_df.columns if c.startswith('days_to_exit_')]
            cols += dte_cols
            display_df = display_df[cols]
            html = display_df.to_html(escape=False, index=False)
            st.markdown('**Liquidity & Float Ownership (selected names)**')
            st.write(html, unsafe_allow_html=True)
            # highlight important flags
            flags = []
            for r in liquidity_rows:
                if r.get('any_single_flag'):
                    flags.append(f"{r['name']} — single manager >{single_manager_threshold:.0f}% of float")
                if r.get('aggregate_flag'):
                    flags.append(f"{r['name']} — aggregate holdings >{aggregate_float_threshold:.0f}% of float")
            if flags:
                st.markdown('**Risk Flags:**')
                for f in flags:
                    st.markdown(f"- {f}")
            # CSV download for liquidity table (raw numeric values)
            try:
                export_df = liq_df.copy()
                # include dynamic DTE columns and adv as numeric
                csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                st.download_button('Download Liquidity CSV', data=csv_bytes, file_name='liquidity_table.csv', mime='text/csv')
            except Exception:
                pass
        else:
            st.write('*No liquidity data available for consensus names.*')

    with tab5:
        st.header('Filings Timeline')
        timeline_rows = []
        for mgr, out in multi_q_by_manager.items():
            for p, df in out:
                timeline_rows.append({'manager':mgr,'period':p,'positions':len(df)})
        st.table(pd.DataFrame(timeline_rows))

    st.markdown('**Note:** This implementation adds multi-quarter parsing, QoQ metrics and signal generation (high conviction new buys, aggressive adds/reductions, full liquidations, new guard). Additional metrics (HHI, crowding score, days-to-exit liquidity tests, and richer visual styling) can be added next.', unsafe_allow_html=True)
