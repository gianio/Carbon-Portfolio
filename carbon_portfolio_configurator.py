# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime # Import datetime for date check
import pytz # For timezone handling
import traceback # For detailed error logging

# ==================================
# Configuration & Theming
# ==================================
st.set_page_config(layout="wide")
css = """
<style>
    /* Main App background - uncomment if desired */
    /* .stApp { background-color: #F1F8E9; } */

    /* --- Sidebar: Green Background & Text --- */
    [data-testid="stSidebar"] { background-color: #C8E6C9; padding-top: 2rem; } /* Lighter Green BG */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-bq {
        color: #1B5E20 !important; /* Dark Green Text/Labels */
    }
    /* --- Sidebar: Green Interactive Elements --- */
    [data-testid="stSidebar"] .stButton>button { background-color: #4CAF50; color: white; border: none; } /* Green Button */
    [data-testid="stSidebar"] .stButton>button:hover { background-color: #388E3C; color: white; } /* Darker Green Hover */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:nth-child(3) { background-color: #388E3C; } /* Dark Green slider handle range */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #66BB6A; color: white; } /* Medium Green Tag */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child[aria-checked="true"] {
        background-color: #388E3C !important; /* Dark Green checkmark background */
    }
    /* --- Sidebar: Consistent Green Theme for other elements --- */
    [data-testid="stSidebar"] .stNumberInput input, [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"],
    [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] {
        border-color: #A5D6A7 !important; /* Light green border for widgets */
    }
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] { background: #A5D6A7; } /* Light green Ticks */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] span:first-child {
        background-color: #A5D6A7 !important; /* Lighter green background for radio */
        border-color: #388E3C !important; /* Dark Green border */
    }

    /* --- Main Content --- */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem; padding-right: 3rem; max-width: 100%; }
    h1, h2, h3, h4, h5, h6 { color: #1B5E20; } /* Titles green */
    .stDataFrame { border: 1px solid #A5D6A7; } /* Dataframe border green */

    /* Style for metric boxes */
    .metric-box {
        border: 2px solid #388E3C; /* Dark Green border */
        border-radius: 5px;
        padding: 10px; /* Adjusted padding */
        margin-bottom: 10px;
        background-color: #E8F5E9; /* Lightest Green background */
        text-align: center;
        font-size: 1.8em; /* Adjusted font size for value */
    }
    .metric-box b { /* Style for the label */
        display: block;
        margin-bottom: 5px;
        color: #1B5E20; /* Dark green label */
        font-size: 0.6em; /* Adjust label font size relative to value */
        font-weight: bold;
    }

    /* Experimental: Increase font size in dataframes */
    .stDataFrame table td, .stDataFrame table th {
        font-size: 115%; /* Adjusted font size */
    }

    /* Styling for the download button */
    div[data-testid="stDownloadButton"] > button {
        background-color: #8ca734 !important; /* Requested Green */
        color: white !important;
        border: none !important; /* Remove border */
        padding: 0.8em 1.5em !important; /* Adjust padding for size */
        width: auto; /* Allow button to size based on content */
        font-size: 1.1em !important; /* Slightly larger font */
        font-weight: bold;
        border-radius: 5px; /* Add rounded corners */
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #6b8e23 !important; /* Darker shade for hover */
        color: white !important;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)
type_color_map = {'technical removal': '#66BB6A', 'natural removal': '#AED581', 'reduction': '#388E3C'}
default_color = '#BDBDBD'

# ==================================
# Allocation Function
# ==================================
def allocate_portfolio(
    project_data: pd.DataFrame, selected_project_names: list, selected_years: list,
    start_year_portfolio: int, end_year_portfolio: int, constraint_type: str, annual_targets: dict,
    removal_target_percent_end_year: float, transition_speed: int, category_split: dict,
    favorite_project: str = None, priority_boost_percent: int = 10,
    min_target_fulfillment_percent: float = 0.95, min_allocation_chunk: int = 1
) -> tuple[dict, pd.DataFrame]:
    portfolio_details = {year: [] for year in selected_years}
    yearly_summary_list = []
    empty_summary_cols = ['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Avg. Price', 'Actual Removal Vol %', 'Target Removal Vol %', 'Total Yearly Margin']

    if not selected_project_names:
        st.warning("No projects selected for allocation.")
        return {}, pd.DataFrame(columns=empty_summary_cols)

    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()
    if project_data_selected.empty:
        st.warning("Selected projects not found in the provided data.")
        return {}, pd.DataFrame(columns=empty_summary_cols)

    required_base_cols = ['project name', 'project type', 'priority']
    price_cols_needed, volume_cols_needed = [], []
    for yr_alloc_setup in selected_years:
        price_cols_needed.append(f"price {yr_alloc_setup}")
        volume_cols_needed.append(f"available volume {yr_alloc_setup}")

    missing_base = [col for col in required_base_cols if col not in project_data_selected.columns]
    if missing_base: raise ValueError(f"Input data missing base columns: {', '.join(missing_base)}")

    missing_years_data = []
    for yr_alloc_check in selected_years:
        if f"price {yr_alloc_check}" not in project_data_selected.columns: missing_years_data.append(f"price {yr_alloc_check}")
        if f"available volume {yr_alloc_check}" not in project_data_selected.columns: missing_years_data.append(f"available volume {yr_alloc_check}")
    if missing_years_data:
        affected_years = sorted(list(set(int(col.split()[-1]) for col in missing_years_data if col.split()[-1].isdigit())))
        raise ValueError(f"Input data missing price/volume for years: {', '.join(map(str, affected_years))}.")

    for col_alloc_num in ['priority'] + price_cols_needed + volume_cols_needed:
        if col_alloc_num in project_data_selected.columns:
            project_data_selected[col_alloc_num] = pd.to_numeric(project_data_selected[col_alloc_num], errors='coerce')
            if col_alloc_num == 'priority': project_data_selected[col_alloc_num] = project_data_selected[col_alloc_num].fillna(0)
            elif col_alloc_num.startswith("available volume"): project_data_selected[col_alloc_num] = project_data_selected[col_alloc_num].fillna(0).apply(lambda x: int(x) if pd.notna(x) and x >= 0 else 0).clip(lower=0)
            elif col_alloc_num.startswith("price"): project_data_selected[col_alloc_num] = project_data_selected[col_alloc_num].fillna(0.0).apply(lambda x: float(x) if pd.notna(x) and x >= 0 else 0.0).clip(lower=0.0)

    all_project_types_in_selection = project_data_selected['project type'].unique()
    is_reduction_selected = 'reduction' in all_project_types_in_selection
    total_years_duration = end_year_portfolio - start_year_portfolio if end_year_portfolio > start_year_portfolio else 0

    for year_loop in selected_years:
        yearly_target_val = annual_targets.get(year_loop, 0)
        price_col_loop = f"price {year_loop}"; volume_col_loop = f"available volume {year_loop}"
        year_total_allocated_vol, year_total_allocated_cost = 0.0, 0.0
        summary_template = {'Year': year_loop, f'Target {constraint_type}': yearly_target_val, 'Allocated Volume': 0.0, 'Allocated Cost': 0.0, 'Avg. Price': 0.0, 'Actual Removal Vol %': 0.0, 'Target Removal Vol %': 0.0, 'Total Yearly Margin': 0.0}

        if yearly_target_val <= 0:
            yearly_summary_list.append(summary_template); portfolio_details[year_loop] = []; continue

        target_percentages = {}
        if is_reduction_selected:
            start_rem_pct, end_rem_pct = 0.10, removal_target_percent_end_year
            progress = 1.0 if total_years_duration <= 0 else max(0, min(1, (year_loop - start_year_portfolio) / total_years_duration if total_years_duration > 0 else 0))
            exponent = 0.1 + (11 - transition_speed) * 0.2; progress_factor = progress ** exponent
            target_rem_pct_yr = start_rem_pct + (end_rem_pct - start_rem_pct) * progress_factor
            target_rem_pct_yr = max(min(start_rem_pct, end_rem_pct), min(max(start_rem_pct, end_rem_pct), target_rem_pct_yr))
            tech_pref, nat_pref = category_split.get('technical removal', 0), category_split.get('natural removal', 0)
            total_rem_pref = tech_pref + nat_pref
            target_tech_rem, target_nat_rem = 0.0, 0.0
            if total_rem_pref > 1e-9:
                target_tech_rem = target_rem_pct_yr * (tech_pref / total_rem_pref)
                target_nat_rem = target_rem_pct_yr * (nat_pref / total_rem_pref)
            elif any(pt in all_project_types_in_selection for pt in ['technical removal', 'natural removal']):
                num_rem_types = sum(1 for pt in ['technical removal', 'natural removal'] if pt in all_project_types_in_selection)
                share = target_rem_pct_yr / num_rem_types if num_rem_types > 0 else 0
                if 'technical removal' in all_project_types_in_selection: target_tech_rem = share
                if 'natural removal' in all_project_types_in_selection: target_nat_rem = share
            target_percentages = {'reduction': max(0.0, 1.0 - target_tech_rem - target_nat_rem), 'technical removal': target_tech_rem, 'natural removal': target_nat_rem}
        else:
            tech_pref, nat_pref = category_split.get('technical removal', 0), category_split.get('natural removal', 0)
            total_pref = tech_pref + nat_pref
            tech_sel, nat_sel = 'technical removal' in all_project_types_in_selection, 'natural removal' in all_project_types_in_selection
            tech_share, nat_share = 0.0, 0.0
            if total_pref > 1e-9 :
                if tech_sel: tech_share = tech_pref / total_pref
                if nat_sel: nat_share = nat_pref / total_pref
            elif tech_sel or nat_sel:
                num_sel_types = tech_sel + nat_sel; base_share = 1.0 / num_sel_types if num_sel_types > 0 else 0.0
                if tech_sel: tech_share = base_share
                if nat_sel: nat_share = base_share
            total_active_share = tech_share + nat_share
            if total_active_share > 1e-9:
                target_percentages['technical removal'] = (tech_share / total_active_share) if tech_sel else 0.0
                target_percentages['natural removal'] = (nat_share / total_active_share) if nat_sel else 0.0
            else: target_percentages['technical removal'], target_percentages['natural removal'] = 0.0, 0.0
            target_percentages['reduction'] = 0.0
        
        current_sum_pct = sum(target_percentages.values())
        if abs(current_sum_pct - 1.0) > 1e-6 and current_sum_pct > 0:
            norm_factor = 1.0 / current_sum_pct
            target_percentages = {ptype: share * norm_factor for ptype, share in target_percentages.items()}
        summary_template['Target Removal Vol %'] = (target_percentages.get('technical removal', 0) + target_percentages.get('natural removal', 0)) * 100

        projects_year_df_loop = project_data_selected[(project_data_selected[price_col_loop] > 0) & (project_data_selected[volume_col_loop] >= min_allocation_chunk)].copy()
        projects_year_df_loop['initial_allocated_volume'] = 0.0
        projects_year_df_loop['initial_allocated_cost'] = 0.0
        projects_year_df_loop['final_priority'] = np.nan

        if projects_year_df_loop.empty:
            yearly_summary_list.append(summary_template); portfolio_details[year_loop] = []; continue

        for project_type_main_loop in all_project_types_in_selection:
            target_share_val = target_percentages.get(project_type_main_loop, 0)
            if target_share_val <= 0: continue
            target_resource_val = yearly_target_val * target_share_val
            projects_of_type_loop = projects_year_df_loop[projects_year_df_loop['project type'] == project_type_main_loop].copy()
            if projects_of_type_loop.empty: continue

            total_prio_type = projects_of_type_loop['priority'].sum()
            projects_of_type_loop['norm_prio_base'] = (1.0 / len(projects_of_type_loop)) if total_prio_type <= 0 and len(projects_of_type_loop) > 0 else (projects_of_type_loop['priority'] / total_prio_type if total_prio_type > 0 else 0)
            current_priorities_dict = projects_of_type_loop.set_index('project name')['norm_prio_base'].to_dict()
            final_priorities_dict = current_priorities_dict.copy()

            is_fav_project_in_type_and_list = favorite_project and \
                                            favorite_project in final_priorities_dict and \
                                            not projects_of_type_loop[projects_of_type_loop['project name'] == favorite_project].empty and \
                                            projects_of_type_loop[projects_of_type_loop['project name'] == favorite_project]['project type'].iloc[0] == project_type_main_loop

            if is_fav_project_in_type_and_list:
                fav_base, boost = current_priorities_dict[favorite_project], priority_boost_percent / 100.0
                increase, new_fav_prio = fav_base * boost, fav_base + fav_base * boost
                others, sum_others = [p for p in current_priorities_dict if p != favorite_project], sum(current_priorities_dict[p_name] for p_name in current_priorities_dict if p_name != favorite_project)
                temp_prios = {favorite_project: new_fav_prio}; reduc_factor = increase / sum_others if sum_others > 1e-9 else 0
                for p_name_other in others: temp_prios[p_name_other] = max(0, current_priorities_dict[p_name_other] * (1 - reduc_factor))
                total_final = sum(temp_prios.values())
                if total_final > 1e-9: final_priorities_dict = {p: prio / total_final for p, prio in temp_prios.items()}
                elif favorite_project in temp_prios : final_priorities_dict = {favorite_project: 1.0}

            project_weights_dict, total_weight_val = {}, 0.0
            if constraint_type == 'Budget':
                for _, r_budget in projects_of_type_loop.iterrows():
                    name_b, final_p_b, price_b = r_budget['project name'], final_priorities_dict.get(r_budget['project name'], 0), r_budget[price_col_loop]
                    project_weights_dict[name_b] = final_p_b * price_b if price_b > 0 else 0; total_weight_val += project_weights_dict[name_b]

            for idx_main_alloc, r_main_alloc in projects_of_type_loop.iterrows():
                name_ma, final_p_ma, avail_vol_ma, price_ma = r_main_alloc['project name'], final_priorities_dict.get(r_main_alloc['project name'], 0), r_main_alloc[volume_col_loop], r_main_alloc[price_col_loop]
                alloc_vol, alloc_cost_val = 0.0, 0.0
                projects_year_df_loop.loc[projects_year_df_loop['project name'] == name_ma, 'final_priority'] = final_p_ma
                if final_p_ma <= 0 or price_ma <= 0 or avail_vol_ma < min_allocation_chunk: continue

                if constraint_type == 'Volume': alloc_vol = min(target_resource_val * final_p_ma, avail_vol_ma)
                elif constraint_type == 'Budget':
                    if total_weight_val > 1e-9: norm_w, budget_proj = project_weights_dict.get(name_ma, 0) / total_weight_val, target_resource_val * (project_weights_dict.get(name_ma, 0) / total_weight_val); alloc_vol = min(budget_proj / price_ma if price_ma > 0 else 0, avail_vol_ma)
                    elif len(projects_of_type_loop) == 1: alloc_vol = min(target_resource_val / price_ma if price_ma > 0 else 0, avail_vol_ma)
                
                alloc_vol_int = int(max(0, math.floor(alloc_vol / min_allocation_chunk) * min_allocation_chunk))
                if alloc_vol_int >= min_allocation_chunk:
                    alloc_cost_val = float(alloc_vol_int * price_ma)
                    idx_update = projects_year_df_loop[projects_year_df_loop['project name'] == name_ma].index[0]
                    projects_year_df_loop.loc[idx_update, 'initial_allocated_volume'] += float(alloc_vol_int)
                    projects_year_df_loop.loc[idx_update, 'initial_allocated_cost'] += alloc_cost_val
                    year_total_allocated_vol += float(alloc_vol_int); year_total_allocated_cost += alloc_cost_val
        
        target_thresh, current_metric = yearly_target_val * min_target_fulfillment_percent, year_total_allocated_cost if constraint_type == 'Budget' else year_total_allocated_vol
        if current_metric < target_thresh and yearly_target_val > 0 :
            needed_val = target_thresh - current_metric
            projects_year_df_loop['remaining_volume'] = projects_year_df_loop[volume_col_loop] - projects_year_df_loop['initial_allocated_volume']
            adj_candidates = projects_year_df_loop[(projects_year_df_loop['remaining_volume'] >= min_allocation_chunk) & (projects_year_df_loop[price_col_loop] > 0)].sort_values(by=['priority', price_col_loop], ascending=[False, True]).copy()
            for _, r_adj in adj_candidates.iterrows():
                if needed_val <= (1e-2 if constraint_type=='Budget' else 1e-6): break
                name_adj_loop, price_adj_loop, avail_adj = r_adj['project name'], r_adj[price_col_loop], r_adj['remaining_volume']
                vol_add_float, cost_add_val = 0.0, 0.0
                add_vol_val = min(avail_adj, needed_val / price_adj_loop if price_adj_loop > 0 else 0) if constraint_type == 'Budget' else min(avail_adj, needed_val)
                add_chunked_int = int(math.floor(add_vol_val / min_allocation_chunk) * min_allocation_chunk)
                if add_chunked_int >= min_allocation_chunk:
                    cost_inc_val = float(add_chunked_int * price_adj_loop)
                    if constraint_type == 'Volume' or (cost_inc_val <= needed_val * 1.1 or cost_inc_val < price_adj_loop * min_allocation_chunk * 1.5):
                        vol_add_float, cost_add_val = float(add_chunked_int), cost_inc_val
                        needed_val -= cost_add_val if constraint_type == 'Budget' else vol_add_float
                if vol_add_float > 0:
                    idx_update_adj = projects_year_df_loop[projects_year_df_loop['project name'] == name_adj_loop].index[0]
                    projects_year_df_loop.loc[idx_update_adj, 'initial_allocated_volume'] += vol_add_float
                    projects_year_df_loop.loc[idx_update_adj, 'initial_allocated_cost'] += cost_add_val
                    year_total_allocated_vol += vol_add_float; year_total_allocated_cost += cost_add_val

        final_alloc_list = []
        final_year_alloc_df_loop = projects_year_df_loop[projects_year_df_loop['initial_allocated_volume'] >= min_allocation_chunk].copy()
        final_year_alloc_df_loop['initial_allocated_volume'] = final_year_alloc_df_loop['initial_allocated_volume'].round().astype(int)

        for _, r_final in final_year_alloc_df_loop.iterrows():
            alloc_vol_final = max(0, r_final['initial_allocated_volume'])
            if 0 < alloc_vol_final < min_allocation_chunk: alloc_vol_final = 0
            if alloc_vol_final >= min_allocation_chunk :
                price_final = r_final.get(price_col_loop, None)
                alloc_cost_final = float(alloc_vol_final * price_final) if price_final is not None else r_final['initial_allocated_cost']
                final_alloc_list.append({'project name': r_final['project name'], 'type': r_final['project type'], 'allocated_volume': alloc_vol_final, 'allocated_cost': alloc_cost_final, 'price_used': price_final, 'priority_applied': r_final['final_priority']})
        
        portfolio_details[year_loop] = final_alloc_list
        year_total_allocated_vol, year_total_allocated_cost = sum(p['allocated_volume'] for p in final_alloc_list), sum(p['allocated_cost'] for p in final_alloc_list)
        summary_template.update({'Allocated Volume': year_total_allocated_vol, 'Allocated Cost': year_total_allocated_cost, 'Avg. Price': (year_total_allocated_cost / year_total_allocated_vol) if year_total_allocated_vol > 0 else 0.0})
        rem_vol_final = sum(p['allocated_volume'] for p in final_alloc_list if p['type'] in ['technical removal', 'natural removal'])
        summary_template['Actual Removal Vol %'] = (rem_vol_final / year_total_allocated_vol * 100) if year_total_allocated_vol > 0 else 0.0
        yearly_summary_list.append(summary_template)

    yearly_summary_df_res = pd.DataFrame(yearly_summary_list)
    if yearly_summary_df_res.empty: yearly_summary_df_res = pd.DataFrame(columns=empty_summary_cols)

    if constraint_type == 'Budget' and not yearly_summary_df_res.empty:
        check_df_budget = yearly_summary_df_res.copy()
        check_df_budget['Target Budget'] = check_df_budget['Year'].map(annual_targets).fillna(0)
        is_overbudget_val = check_df_budget['Allocated Cost'] > check_df_budget['Target Budget'] * 1.001
        overbudget_df = check_df_budget[is_overbudget_val]
        if not overbudget_df.empty: st.warning(f"Budget slightly exceeded: {overbudget_df['Year'].tolist()}")
    return portfolio_details, yearly_summary_df_res

# ==================================
# Margin Calculation Functions
# ==================================
def get_margin_per_unit(project_data_row: pd.Series, allocated_price_this_year: float) -> float:
    if pd.isna(allocated_price_this_year) or allocated_price_this_year < 0: return 0.0
    base_price_col, thresh_price_col, mshare_col, fpp_col, pms_col = 'base price', 'threshold price', 'margin share', 'fixed purchase price', 'percental margin share'
    margin_pu = 0.0
    bp, tp, ms = project_data_row.get(base_price_col), project_data_row.get(thresh_price_col), project_data_row.get(mshare_col)
    try:
        if pd.notna(bp) and pd.notna(tp) and pd.notna(ms): margin_pu = max(0, allocated_price_this_year - float(tp)) * float(ms) + (float(tp) - float(bp))
        elif pd.notna(project_data_row.get(fpp_col)): margin_pu = allocated_price_this_year - float(project_data_row.get(fpp_col))
        elif pd.notna(project_data_row.get(pms_col)): margin_pu = allocated_price_this_year * float(project_data_row.get(pms_col))
    except: margin_pu = 0.0
    return margin_pu if pd.notna(margin_pu) and margin_pu != -np.inf else 0.0

def add_margins_to_details_df(details_df: pd.DataFrame, project_master_data: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        expected_cols = list(details_df.columns)
        if 'margin_per_unit' not in expected_cols: expected_cols.append('margin_per_unit')
        if 'margin' not in expected_cols: expected_cols.append('margin')
        return pd.DataFrame(columns=expected_cols)
        
    rows_with_margins = []
    if 'project name' not in project_master_data.columns:
        st.error("DEBUG: 'project name' not in project_master_data for margin calculation.")
        temp_df_error = details_df.copy()
        temp_df_error['margin_per_unit'], temp_df_error['margin'] = 0.0, 0.0
        return temp_df_error
        
    project_lookup_margin = project_master_data.set_index('project name')
    for _, row_margin_calc in details_df.iterrows():
        proj_name, alloc_vol, price_used = row_margin_calc['project name'], row_margin_calc['volume'], row_margin_calc['price']
        new_row_data = row_margin_calc.to_dict()
        margin_total, margin_pu_val = 0.0, 0.0
        if proj_name in project_lookup_margin.index and pd.notna(price_used) and alloc_vol > 0:
            proj_data_row_margin = project_lookup_margin.loc[proj_name]
            if isinstance(proj_data_row_margin, pd.DataFrame): proj_data_row_margin = proj_data_row_margin.iloc[0]
            margin_pu_val = get_margin_per_unit(proj_data_row_margin, price_used)
            margin_total = margin_pu_val * alloc_vol
        new_row_data.update({'margin_per_unit': margin_pu_val, 'margin': margin_total})
        rows_with_margins.append(new_row_data)
    
    if not rows_with_margins and not details_df.empty: 
        temp_df = details_df.copy()
        temp_df['margin_per_unit'], temp_df['margin'] = 0.0, 0.0
        return temp_df
    return pd.DataFrame(rows_with_margins)

# ==================================
# Streamlit App Layout & Logic
# ==================================
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar", help="Required: project name, project type, priority, price_YYYY, available_volume_YYYY. Optional margins: base price, threshold price, margin share, fixed purchase price, percental margin share. Optional: description, project_link.")
    default_values = {'working_data_full': None, 'selected_years': [], 'selected_projects': [], 'project_names': [], 'favorite_projects_selection': [], 'actual_start_year': None, 'actual_end_year': None, 'available_years_in_data': [], 'constraint_type': 'Volume', 'removal_target_end_year': 0.8, 'transition_speed': 5, 'category_split': {'technical removal': 0.5, 'natural removal': 0.5}, 'annual_targets': {}, 'master_target': None, 'data_loaded_successfully': False, 'years_slider_sidebar': 5, 'min_fulfillment_perc': 95, 'removal_preference_slider': 5, 'min_alloc_chunk': 1}
    for key, default_value in default_values.items():
        if key not in st.session_state: st.session_state[key] = default_value

    if df_upload:
        @st.cache_data
        def load_and_prepare_data(uploaded_file):
            try:
                data = pd.read_csv(uploaded_file, sep=';', encoding='utf-8-sig')
                data.columns = data.columns.str.lower().str.strip().str.replace(r'\s+', '_', regex=True)
                standardized_columns_found = data.columns.tolist()
            except Exception as read_error:
                try:
                    if hasattr(uploaded_file, 'seek'): uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, sep=';')
                    data.columns = data.columns.str.lower().str.strip().str.replace(r'\s+', '_', regex=True)
                    standardized_columns_found = data.columns.tolist()
                except Exception as read_error_fallback:
                    return None, f"Error reading/processing CSV (delimiter=';'): {read_error} / {read_error_fallback}", [], [], []

            core_cols_std = ['project_name', 'project_type', 'priority']
            optional_cols_std = ['description', 'project_link']
            margin_cols_std = ['base_price', 'threshold_price', 'margin_share', 'fixed_purchase_price', 'percental_margin_share']

            missing_essential = [] 
            for core_val_check in core_cols_std:
                is_present_flag_check = False
                for found_val_check in standardized_columns_found:
                    if core_val_check == found_val_check: 
                        is_present_flag_check = True; break
                if not is_present_flag_check: missing_essential.append(core_val_check)

            if missing_essential:
                found_cols_str = ", ".join(standardized_columns_found)
                return None, f"CSV missing essential: {', '.join(missing_essential)}. Expected after std. FOUND: [{found_cols_str}]. Check CSV headers & delimiter (';').", [], [], []

            if 'treshold_price' in data.columns and 'threshold_price' not in data.columns:
                data.rename(columns={'treshold_price': 'threshold_price'}, inplace=True)
            
            for m_col in margin_cols_std:
                if m_col not in data.columns: data[m_col] = np.nan
            
            numeric_prefixes_std = ['price_', 'available_volume_']
            cols_to_convert_numeric = ['priority'] + margin_cols_std
            available_years_set, year_data_cols_found_list = set(), []

            for col_scan in data.columns:
                for prefix_scan in numeric_prefixes_std:
                    if col_scan.startswith(prefix_scan) and col_scan[len(prefix_scan):].isdigit():
                        cols_to_convert_numeric.append(col_scan); year_data_cols_found_list.append(col_scan)
                        available_years_set.add(int(col_scan[len(prefix_scan):])); break
            
            if not available_years_set: st.sidebar.warning("No 'price_YYYY'/'volume_YYYY' columns found.")

            for col_num_conv in list(set(cols_to_convert_numeric)):
                if col_num_conv in data.columns: data[col_num_conv] = pd.to_numeric(data[col_num_conv], errors='coerce')

            data['priority'] = data['priority'].fillna(0).clip(lower=0)
            for m_col_proc in margin_cols_std:
                 if m_col_proc in data.columns and m_col_proc in ['base_price', 'threshold_price', 'fixed_purchase_price']: data[m_col_proc] = data[m_col_proc].apply(lambda x: x if pd.notna(x) and x >= 0 else np.nan)

            for col_yr_proc in data.columns:
                if col_yr_proc.startswith('available_volume_') and col_yr_proc in year_data_cols_found_list: data[col_yr_proc] = data[col_yr_proc].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0)
                elif col_yr_proc.startswith('price_') and col_yr_proc in year_data_cols_found_list: data[col_yr_proc] = data[col_yr_proc].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0)
            
            available_years_list = sorted(list(available_years_set))
            invalid_types_list = []
            if 'project_type' in data.columns:
                data['project_type'] = data['project_type'].astype(str).str.lower().str.strip()
                valid_types = ['reduction', 'technical removal', 'natural removal']
                invalid_df = data[~data['project_type'].isin(valid_types)]
                if not invalid_df.empty:
                    invalid_types_list = invalid_df['project_type'].unique().tolist()
                    data = data[data['project_type'].isin(valid_types)].copy()
            
            # Define all columns that are expected and should be kept if they exist
            # These names are all standardized (e.g., 'project_name', 'price_2025')
            defined_cols_to_process = list(set(core_cols_std + margin_cols_std + optional_cols_std + year_data_cols_found_list))
            # Filter `data` to only include columns that are in `defined_cols_to_process` AND currently exist in `data.columns`
            # This ensures that only known, processed columns are passed forward.
            actual_cols_to_keep = [col for col in data.columns if col in defined_cols_to_process]
            data = data[actual_cols_to_keep]


            final_rename_map_dict = {'project_name': 'project name', 'project_type': 'project type', 'priority': 'priority', 
                                     'description': 'Description', 'project_link': 'Project Link', 
                                     'base_price': 'base price', 'threshold_price': 'threshold price', 
                                     'margin_share': 'margin share', 'fixed_purchase_price': 'fixed purchase price', 
                                     'percental_margin_share': 'percental margin share'}
            for yr_col_map in year_data_cols_found_list: # These are standardized, e.g. price_2025
                final_rename_map_dict[yr_col_map] = yr_col_map.replace('_', ' ') # price_2025 -> price 2025
            
            # Apply renaming only for columns that exist in `data` (which has standardized names now)
            actual_rename_map = {k_std: v_disp for k_std, v_disp in final_rename_map_dict.items() if k_std in data.columns}
            data.rename(columns=actual_rename_map, inplace=True)

            project_names_output = sorted(data['project name'].unique().tolist()) if 'project name' in data.columns else [] # Use display name
            return data, None, available_years_list, project_names_output, invalid_types_list

        try: # Calling load_and_prepare_data
            data_main, err_msg, years_data, proj_names, invalid_types = load_and_prepare_data(df_upload)
            if invalid_types: st.sidebar.warning(f"Ignored invalid project types: {', '.join(invalid_types)}.")
            if err_msg: st.sidebar.error(err_msg); st.session_state.update({'data_loaded_successfully': False, 'working_data_full': None, 'project_names': [], 'available_years_in_data': [], 'selected_projects': [], 'annual_targets': {}})
            else:
                st.session_state.update({'project_names': proj_names, 'available_years_in_data': years_data, 'working_data_full': data_main, 'data_loaded_successfully': True, 'annual_targets': {}})
                st.sidebar.success("Data loaded successfully!")
                curr_sel, valid_sel = st.session_state.get('selected_projects', []), [p for p in st.session_state.get('selected_projects', []) if p in proj_names]
                st.session_state.selected_projects = valid_sel if valid_sel or not proj_names else proj_names
        except Exception as e: st.sidebar.error(f"File processing error: {e}"); st.sidebar.error(f"Traceback: {traceback.format_exc()}"); st.session_state.update({'data_loaded_successfully': False, 'working_data_full': None})

    if st.session_state.get('data_loaded_successfully', False):
        data_ui, years_ui, names_ui = st.session_state.working_data_full, st.session_state.available_years_in_data, st.session_state.project_names
        if not years_ui: st.sidebar.warning("No usable year data. Settings disabled.")
        else:
            st.markdown("## 2. Portfolio Settings")
            min_yr_sb, max_yr_sb = min(years_ui), max(years_ui)
            max_yrs_plan_sb = max(1, max_yr_sb - min_yr_sb + 1)
            curr_yrs_plan_val_sb = st.session_state.get('years_slider_sidebar', 5); 
            try: curr_yrs_plan_val_sb = int(curr_yrs_plan_val_sb)
            except: curr_yrs_plan_val_sb = 5
            curr_yrs_plan_val_sb = max(1, min(curr_yrs_plan_val_sb, max_yrs_plan_sb))
            yrs_to_plan_sb = st.number_input(f"Years to Plan (Starting {min_yr_sb})", 1, max_yrs_plan_sb, curr_yrs_plan_val_sb, 1, key='years_slider_sidebar_widget', help=f"Enter # years (1 to {max_yrs_plan_sb}).")
            st.session_state.years_slider_sidebar = yrs_to_plan_sb
            start_yr_sel_sb, end_yr_sel_sb = min_yr_sb, min_yr_sb + yrs_to_plan_sb - 1
            st.session_state.selected_years = [yr for yr in range(start_yr_sel_sb, end_yr_sel_sb + 1) if f"price {yr}" in data_ui.columns and f"available volume {yr}" in data_ui.columns]
            if not st.session_state.selected_years:
                st.sidebar.error(f"No data for period ({start_yr_sel_sb}-{end_yr_sel_sb}). Adjust 'Years to Plan'.")
                st.session_state.actual_start_year, st.session_state.actual_end_year = None, None
            else:
                st.session_state.actual_start_year, st.session_state.actual_end_year = min(st.session_state.selected_years), max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")
                st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar_widget', horizontal=True)
                st.markdown("### Annual Target Settings")
                is_budget_sb = st.session_state.constraint_type == 'Budget'
                master_target_sb = st.session_state.get('master_target')
                default_master_val_sb = 100000.0 if is_budget_sb else 1000
                if master_target_sb is not None:
                    try: default_master_val_sb = float(master_target_sb) if is_budget_sb else int(float(master_target_sb))
                    except: pass
                num_args_master_sb = {"min_value": 0.0 if is_budget_sb else 0, "step": 1000.0 if is_budget_sb else 100, "value": default_master_val_sb, "key": f'master_{st.session_state.constraint_type.lower()}_sidebar', "help": "Set default. Override below."}
                if is_budget_sb: num_args_master_sb["format"] = "%.2f"
                st.session_state.master_target = st.number_input(f"Default Annual Target ({'€' if is_budget_sb else 't'}):", **num_args_master_sb)
                with st.expander("Customize Annual Targets (+/-)", expanded=False):
                    updated_annual_targets_sb = {} 
                    if not st.session_state.selected_years: st.caption("Select years first.")
                    else:
                        for yr_target_sb_loop in st.session_state.selected_years: 
                            yr_target_val_sb_loop = st.session_state.get('annual_targets', {}).get(yr_target_sb_loop, st.session_state.master_target)
                            key_sb_loop = f"target_{yr_target_sb_loop}_{st.session_state.constraint_type}"
                            label_sb_loop = f"Target {yr_target_sb_loop} ({'€' if is_budget_sb else 't'})"
                            num_args_yr_sb_loop = {"min_value": 0.0 if is_budget_sb else 0, "step": 1000.0 if is_budget_sb else 100, "key": key_sb_loop}
                            try: num_args_yr_sb_loop["value"] = float(yr_target_val_sb_loop) if is_budget_sb else int(float(yr_target_val_sb_loop))
                            except: num_args_yr_sb_loop["value"] = float(st.session_state.master_target) if is_budget_sb else int(st.session_state.master_target)
                            if is_budget_sb: num_args_yr_sb_loop["format"] = "%.2f"
                            updated_annual_targets_sb[yr_target_sb_loop] = st.number_input(label_sb_loop, **num_args_yr_sb_loop)
                    st.session_state.annual_targets = updated_annual_targets_sb
                st.sidebar.markdown("### Allocation Goal & Preferences")
                st.session_state.min_fulfillment_perc = st.sidebar.slider(f"Min. Target Fulfillment (%)", 50, 100, st.session_state.get('min_fulfillment_perc', 95), key='min_fulfill_perc_sidebar')
                st.session_state.min_alloc_chunk = int(st.sidebar.number_input("Min. Allocation Unit (t)", 1, step=1, value=st.session_state.get('min_alloc_chunk', 1), key='min_alloc_chunk_sidebar') or 1)
                st.sidebar.markdown("### Removal Volume Transition (If 'Reduction' Projects Used)")
                reduc_present_sb_val = 'reduction' in data_ui['project type'].unique() if 'project type' in data_ui.columns else False
                st.sidebar.info("Applies if 'Reduction' projects selected." if reduc_present_sb_val else "Inactive: No 'Reduction' projects.")
                end_yr_help_sb_val = st.session_state.actual_end_year or "end year"
                try: rem_target_default_sb_val = int(float(st.session_state.get('removal_target_end_year', 0.8)) * 100)
                except: rem_target_default_sb_val = 80
                st.session_state.removal_target_end_year = st.sidebar.slider(f"Target Removal Vol % ({end_yr_help_sb_val})", 0, 100, rem_target_default_sb_val, key='removal_perc_slider_sidebar', disabled=not reduc_present_sb_val) / 100.0
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), key='transition_speed_slider_sidebar', disabled=not reduc_present_sb_val)
                st.sidebar.markdown("### Removal Category Preference")
                rem_types_present_sb_val = any(pt in data_ui['project type'].unique() for pt in ['technical removal', 'natural removal']) if 'project type' in data_ui.columns else False
                rem_pref_val_sb_val = st.sidebar.slider("Technical vs Natural Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d", key='removal_pref_slider_sidebar', disabled=not rem_types_present_sb_val)
                st.session_state['removal_preference_slider'] = rem_pref_val_sb_val; tech_pref_ratio_sb_val = (rem_pref_val_sb_val - 1) / 9.0
                st.session_state.category_split = {'technical removal': tech_pref_ratio_sb_val, 'natural removal': 1.0 - tech_pref_ratio_sb_val}
                st.sidebar.markdown("## 3. Select Projects")
                if not names_ui: st.sidebar.warning("No projects available.")
                else:
                    st.session_state.selected_projects = st.sidebar.multiselect("Select projects to include:", options=names_ui, default=st.session_state.get('selected_projects', names_ui), key='project_selector_sidebar')
                    if 'priority' in data_ui.columns:
                        boost_opts_sb_val = [p for p in names_ui if p in st.session_state.selected_projects]
                        if boost_opts_sb_val:
                            fav_default_sb_val = [f for f in st.session_state.get('favorite_projects_selection', []) if f in boost_opts_sb_val][:1]
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect("Favorite Project (Priority Boost):", options=boost_opts_sb_val, default=fav_default_sb_val, key='favorite_selector_sidebar', max_selections=1)
                        else: st.session_state.favorite_projects_selection = []
                    else: st.session_state.favorite_projects_selection = []

# ==================================
# Main Page Content
# ==================================
st.markdown(f"<h1 style='color: #8ca734;'>Carbon Portfolio Builder</h1>", unsafe_allow_html=True)
st.markdown("---")

if not st.session_state.get('data_loaded_successfully', False):
    st.info("👋 Welcome! Please upload your project data CSV using the sidebar menu to get started.")
elif st.session_state.get('data_loaded_successfully', False):
    st.markdown("## Project Offerings")
    st.caption("To get more information on the specific project please unfold the list and click the link to the project slide.")
    project_df_disp = st.session_state.working_data_full.copy()
    avg_price_cols_disp = [f"price {yr}" for yr in st.session_state.available_years_in_data if f"price {yr}" in project_df_disp.columns]
    project_df_disp['Average Price'] = project_df_disp[avg_price_cols_disp].mean(axis=1, skipna=True).fillna(0.0) if avg_price_cols_disp else 0.0
    with st.expander("View Project Details"):
        if not project_df_disp.empty:
            cols_proj_disp = ['project name', 'project type', 'Average Price']
            if 'Project Link' in project_df_disp.columns: cols_proj_disp.append('Project Link')
            col_config_disp = {"Average Price": st.column_config.NumberColumn("Avg Price (€/t)", format="€%.2f")}
            if 'Project Link' in project_df_disp.columns: col_config_disp["Project Link"] = st.column_config.LinkColumn("Project Link", display_text="Visit ->")
            st.dataframe(project_df_disp[cols_proj_disp], column_config=col_config_disp, hide_index=True, use_container_width=True)
        else: st.write("No project data to display.")
    st.markdown("---")

if st.session_state.get('data_loaded_successfully', False):
    if not st.session_state.get('selected_projects'): st.warning("⚠️ Please select projects in the sidebar (Section 3).")
    elif not st.session_state.get('selected_years'): st.warning("⚠️ No valid years for planning. Adjust 'Years to Plan' (Section 2).")
    else:
        required_keys_main = ['working_data_full', 'selected_projects', 'selected_years', 'actual_start_year', 'actual_end_year', 'constraint_type', 'annual_targets', 'removal_target_end_year', 'transition_speed', 'category_split', 'favorite_projects_selection', 'min_fulfillment_perc', 'min_alloc_chunk']
        keys_ok = all(st.session_state.get(k) is not None for k in required_keys_main if k not in ['annual_targets', 'favorite_projects_selection']) and \
                  isinstance(st.session_state.get('annual_targets'), dict) and \
                  isinstance(st.session_state.get('favorite_projects_selection'), list)

        if keys_ok:
            try:
                fav_proj_run = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
                annual_targets_run = st.session_state.annual_targets
                if not annual_targets_run and st.session_state.selected_years:
                    annual_targets_run = {yr: 0 for yr in st.session_state.selected_years}

                if st.session_state.constraint_type == 'Budget': st.info(f"**Budget Mode:** Projects get budget via weighted priority & price...")
                st.success(f"**Allocation Goal:** Attempting ≥ **{st.session_state.min_fulfillment_perc}%** of annual target {st.session_state.constraint_type}.")
                
                with st.spinner("Calculating portfolio..."):
                    results_run, summary_df_run = allocate_portfolio(
                        project_data=st.session_state.working_data_full, selected_project_names=st.session_state.selected_projects,
                        selected_years=st.session_state.selected_years, start_year_portfolio=st.session_state.actual_start_year,
                        end_year_portfolio=st.session_state.actual_end_year, constraint_type=st.session_state.constraint_type,
                        annual_targets=annual_targets_run, removal_target_percent_end_year=st.session_state.removal_target_end_year,
                        transition_speed=st.session_state.transition_speed, category_split=st.session_state.category_split,
                        favorite_project=fav_proj_run, min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0,
                        min_allocation_chunk=st.session_state.min_alloc_chunk)
                
                details_list_run = []
                if results_run:
                    for yr_res, projs_res in results_run.items():
                        if projs_res:
                            for proj_detail in projs_res:
                                if isinstance(proj_detail, dict) and (proj_detail.get('allocated_volume', 0) >= st.session_state.min_alloc_chunk or proj_detail.get('allocated_cost', 0) > 1e-6):
                                    details_list_run.append({'year': yr_res, 'project name': proj_detail.get('project name'), 'type': proj_detail.get('type'), 'volume': proj_detail.get('allocated_volume', 0), 'price': proj_detail.get('price_used', None), 'cost': proj_detail.get('allocated_cost', 0.0)})
                
                details_df_run = pd.DataFrame(details_list_run)
                total_portfolio_margin_run = 0.0
                details_df_with_margins_run = pd.DataFrame()

                if not details_df_run.empty:
                    details_df_with_margins_run = add_margins_to_details_df(details_df_run.copy(), st.session_state.working_data_full)
                    if not details_df_with_margins_run.empty and 'margin' in details_df_with_margins_run.columns:
                        total_portfolio_margin_run = details_df_with_margins_run['margin'].sum()
                
                if 'Total Yearly Margin' not in summary_df_run.columns: summary_df_run['Total Yearly Margin'] = 0.0 
                if not summary_df_run.empty:
                    if not details_df_with_margins_run.empty and 'year' in details_df_with_margins_run.columns and 'margin' in details_df_with_margins_run.columns and not details_df_with_margins_run.groupby('year')['margin'].sum().empty:
                        actual_yearly_margins_series = details_df_with_margins_run.groupby('year')['margin'].sum()
                        summary_df_run['Total Yearly Margin'] = summary_df_run['Year'].map(actual_yearly_margins_series).fillna(0.0)
                    else: summary_df_run['Total Yearly Margin'] = 0.0 

                st.markdown("## Portfolio Summary"); colL, colM, colR = st.columns([1.5, 1.5, 1.2], gap="large")
                with colL:
                    st.markdown("#### Key Metrics (Overall)")
                    tot_cost, tot_vol, avg_price_overall = 0.0, 0, 0.0
                    if not summary_df_run.empty:
                        tot_cost, tot_vol = summary_df_run['Allocated Cost'].sum(), summary_df_run['Allocated Volume'].sum()
                        avg_price_overall = tot_cost / tot_vol if tot_vol > 0 else 0.0
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> € {tot_cost:,.2f}</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> {tot_vol:,.0f} t</div>""", unsafe_allow_html=True)
                with colM:
                    st.markdown("#### &nbsp;")
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Margin</b> € {total_portfolio_margin_run:,.2f}</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> € {avg_price_overall:,.2f} /t</div>""", unsafe_allow_html=True)
                with colR:
                    st.markdown("#### Volume by Project Type")
                    df_for_pie = details_df_with_margins_run if not details_df_with_margins_run.empty and 'volume' in details_df_with_margins_run.columns else details_df_run
                    if not df_for_pie.empty and 'volume' in df_for_pie.columns and df_for_pie['volume'].sum() > 1e-6:
                        pie_data_run = df_for_pie.groupby('type')['volume'].sum().reset_index(name='volume')
                        pie_data_run = pie_data_run[pie_data_run['volume'] > 1e-6]
                        if not pie_data_run.empty:
                            fig_pie_run = px.pie(pie_data_run, values='volume', names='type', color='type', color_discrete_map=type_color_map, height=350)
                            fig_pie_run.update_layout(legend_title_text='Project Type', legend_orientation="h", legend_yanchor="bottom", legend_y=-0.2, legend_xanchor="center", legend_x=0.5, margin=dict(t=5, b=50, l=0, r=0))
                            fig_pie_run.update_traces(textposition='inside', textinfo='percent', sort=False, hole=.3, marker=dict(line=dict(color='#FFFFFF', width=1)))
                            st.plotly_chart(fig_pie_run, use_container_width=True)
                        else: st.caption("No significant volume for pie chart.")
                    else: st.caption("No allocation details for pie chart.")
                
                st.markdown("---")
                df_for_plots_tables = details_df_with_margins_run if not details_df_with_margins_run.empty and 'margin' in details_df_with_margins_run.columns else details_df_run.assign(margin=0.0)

                if df_for_plots_tables.empty : st.warning("No detailed project allocations for plots/tables.")
                else:
                    st.markdown("### Portfolio Composition & Price Over Time")
                    if 'year' in df_for_plots_tables.columns: df_for_plots_tables['year'] = df_for_plots_tables['year'].astype(int)
                    summary_plot_data_run = df_for_plots_tables.groupby(['year', 'type'], as_index=False).agg(volume=('volume', 'sum'), cost=('cost', 'sum'), margin=('margin', 'sum'))
                    if 'margin' not in summary_plot_data_run.columns: summary_plot_data_run['margin'] = 0.0
                    
                    price_summary_data_run = summary_df_run[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'}) if not summary_df_run.empty else pd.DataFrame(columns=['year', 'avg_price'])
                    fig_comp_run = make_subplots(specs=[[{"secondary_y": True}]])
                    y_metric_plot = 'volume' if st.session_state.constraint_type == 'Volume' else 'cost'
                    y_label_plot = f"Allocated {y_metric_plot.capitalize()} ({'t' if y_metric_plot == 'volume' else '€'})"
                    y_hover_plot_template = f"{y_metric_plot.capitalize()}: %{{y:{'{:,.0f}' if y_metric_plot == 'volume' else '€{:,.2f}'}}}<extra></extra>"
                    for t_name_plot_loop in ['reduction', 'natural removal', 'technical removal']:
                        if t_name_plot_loop in summary_plot_data_run['type'].unique():
                            df_type_plot_loop = summary_plot_data_run[summary_plot_data_run['type'] == t_name_plot_loop]
                            if not df_type_plot_loop.empty and y_metric_plot in df_type_plot_loop and df_type_plot_loop[y_metric_plot].sum() > 1e-6 :
                                fig_comp_run.add_trace(go.Bar(x=df_type_plot_loop['year'], y=df_type_plot_loop[y_metric_plot], name=t_name_plot_loop.replace('_', ' ').capitalize(), marker_color=type_color_map.get(t_name_plot_loop, default_color), hovertemplate=f'Year: %{{x}}<br>Type: {t_name_plot_loop.replace("_", " ").capitalize()}<br>{y_hover_plot_template}'), secondary_y=False)
                    if not price_summary_data_run.empty: fig_comp_run.add_trace(go.Scatter(x=price_summary_data_run['year'], y=price_summary_data_run['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker_symbol='circle', marker_size=8, line={"color":'#1B5E20', "width":3}, hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}/t<extra></extra>'), secondary_y=True)
                    if not summary_df_run.empty and 'Actual Removal Vol %' in summary_df_run.columns: fig_comp_run.add_trace(go.Scatter(x=summary_df_run['Year'], y=summary_df_run['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker_symbol='star', marker_size=8, hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                    y2_max_plot_val = 105.0
                    if not price_summary_data_run.empty and 'avg_price' in price_summary_data_run.columns and price_summary_data_run['avg_price'].notna().any(): y2_max_plot_val = max(y2_max_plot_val, price_summary_data_run['avg_price'].max() * 1.1 if pd.notna(price_summary_data_run['avg_price'].max()) else y2_max_plot_val)
                    if not summary_df_run.empty and 'Actual Removal Vol %' in summary_df_run.columns and summary_df_run['Actual Removal Vol %'].notna().any(): y2_max_plot_val = max(y2_max_plot_val, summary_df_run['Actual Removal Vol %'].max() * 1.1 if pd.notna(summary_df_run['Actual Removal Vol %'].max()) else y2_max_plot_val)
                    fig_comp_run.update_layout(xaxis_title='Year', yaxis_title=y_label_plot, yaxis2_title='Avg Price (€/t) / Rem. %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis_rangemode='tozero', yaxis2=dict(rangemode='tozero', range=[0, y2_max_plot_val]), hovermode="x unified")
                    if st.session_state.selected_years: fig_comp_run.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                    st.plotly_chart(fig_comp_run, use_container_width=True)

                    st.markdown("### Detailed Allocation by Project and Year")
                    pivot_display_run = pd.DataFrame()
                    df_for_pivot_main = df_for_plots_tables 
                    if 'margin' not in df_for_pivot_main.columns : df_for_pivot_main['margin'] = 0.0

                    if not df_for_pivot_main.empty and 'year' in df_for_pivot_main.columns:
                        try:
                            df_for_pivot_main['year'] = pd.to_numeric(df_for_pivot_main['year'])
                            pivot_intermediate_run = pd.pivot_table(df_for_pivot_main, values=['volume', 'cost', 'price', 'margin'], index=['project name', 'type'], columns='year', aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first', 'margin': 'sum'})
                            
                            pivot_final_run = pd.DataFrame() 
                            yrs_pivot_run_list = [] 
                            if not pivot_intermediate_run.empty:
                                pivot_final_run = pivot_intermediate_run.swaplevel(0, 1, axis=1)
                                metric_order_run = ['volume', 'cost', 'price', 'margin']
                                yrs_pivot_run_list = sorted([yr for yr in pivot_final_run.columns.get_level_values(0).unique() if isinstance(yr, (int, float, np.number))])
                                
                                if yrs_pivot_run_list:
                                    final_multi_idx_run = pd.MultiIndex.from_product([yrs_pivot_run_list, metric_order_run], names=['year', 'metric'])
                                    pivot_final_run = pivot_final_run.reindex(columns=final_multi_idx_run).sort_index(axis=1, level=[0, 1])
                                else: pivot_final_run = pd.DataFrame(index=pivot_intermediate_run.index)
                                pivot_final_run.index.names = ['Project Name', 'Type']
                            pivot_display_run = pivot_final_run.copy()

                            if not summary_df_run.empty and 'Year' in summary_df_run.columns and yrs_pivot_run_list:
                                total_data_dict_run = {}
                                summary_indexed_run = summary_df_run.set_index('Year')
                                for yr_total_run_loop in yrs_pivot_run_list:
                                    if yr_total_run_loop in summary_indexed_run.index:
                                        s_row_run = summary_indexed_run.loc[yr_total_run_loop]
                                        total_data_dict_run[(yr_total_run_loop, 'volume')] = s_row_run.get('Allocated Volume',0)
                                        total_data_dict_run[(yr_total_run_loop, 'cost')] = s_row_run.get('Allocated Cost',0.0)
                                        total_data_dict_run[(yr_total_run_loop, 'price')] = s_row_run.get('Avg. Price',0.0)
                                        total_data_dict_run[(yr_total_run_loop, 'margin')] = s_row_run.get('Total Yearly Margin',0.0)
                                
                                if total_data_dict_run :
                                    total_row_df_run = pd.DataFrame(total_data_dict_run, index=pd.MultiIndex.from_tuples([('Total Portfolio', 'All Types')], names=['Project Name', 'Type']))
                                    if isinstance(total_row_df_run.columns, pd.MultiIndex): total_row_df_run.columns.names = ['year', 'metric']
                                    pivot_display_run = pd.concat([pivot_display_run, total_row_df_run]) if not pivot_display_run.empty else total_row_df_run
                            
                            # --- MODIFIED SECTION WITH MORE ROBUST ERROR HANDLING FOR .xs ---
                            if not pivot_display_run.empty:
                                # Proactively create 'Total Margin' column, initialized to 0.0
                                pivot_display_run['Total Margin'] = 0.0
                                try:
                                    # Attempt to set column level names robustly
                                    if isinstance(pivot_display_run.columns, pd.MultiIndex) and pivot_display_run.columns.nlevels == 2:
                                        pivot_display_run.columns.names = ['year', 'metric']
                                    
                                    has_margin_metric_and_level = False
                                    if isinstance(pivot_display_run.columns, pd.MultiIndex) and \
                                       'metric' in pivot_display_run.columns.names: # Check level name exists
                                        # Check if 'margin' is actually a value in the 'metric' level
                                        if 'margin' in pivot_display_run.columns.get_level_values('metric'):
                                            has_margin_metric_and_level = True
                                    
                                    if has_margin_metric_and_level:
                                        margin_sum = pivot_display_run.xs('margin', axis=1, level='metric').sum(axis=1)
                                        pivot_display_run['Total Margin'] = margin_sum
                                    else:
                                        st.info("Detailed Table: 'margin' data or expected column structure not found for 'Total Margin' sum. Values set to 0.")
                                        # 'Total Margin' column already exists and is 0.0, so no action needed here.
                                
                                except KeyError as e_xs: 
                                    st.warning(f"Could not calculate 'Total Margin' for detailed table due to KeyError ('{e_xs}', likely level 'metric' not found). Setting to 0.")
                                    # 'Total Margin' column already exists and is 0.0
                                except Exception as e_general_margin_calc:
                                    st.error(f"An unexpected error occurred while calculating 'Total Margin' for detailed table: {e_general_margin_calc}")
                                    # 'Total Margin' column already exists and is 0.0
                                
                                pivot_display_run = pivot_display_run.fillna(0)
                                formatter_run = {}
                                for col_tuple_fmt_loop in pivot_display_run.columns:
                                    if isinstance(col_tuple_fmt_loop, tuple) and len(col_tuple_fmt_loop) == 2:
                                        metric_fmt_val = col_tuple_fmt_loop[1]
                                        if metric_fmt_val == 'volume': formatter_run[col_tuple_fmt_loop] = '{:,.0f} t'
                                        elif metric_fmt_val == 'cost': formatter_run[col_tuple_fmt_loop] = '€{:,.2f}'
                                        elif metric_fmt_val == 'price': formatter_run[col_tuple_fmt_loop] = lambda x_fmt_val: f'€{x_fmt_val:,.2f}/t' if pd.notna(x_fmt_val) and x_fmt_val != 0 else '-'
                                        elif metric_fmt_val == 'margin': formatter_run[col_tuple_fmt_loop] = '€{:,.2f}'
                                    elif col_tuple_fmt_loop == 'Total Margin': formatter_run[col_tuple_fmt_loop] = '€{:,.2f}'
                                st.dataframe(pivot_display_run.style.format(formatter_run, na_rep="-"), use_container_width=True)
                            # --- END MODIFIED SECTION ---
                            else: st.info("No data for detailed allocation table.")
                        except Exception as e_pivot_run_main: st.error(f"Could not create detailed table: {e_pivot_run_main}"); st.error(f"Traceback: {traceback.format_exc()}")
                    else: st.info("No allocation details for detailed table (or 'year' column missing).")
                    
                    if not pivot_display_run.empty:
                        csv_df_export = pivot_display_run.copy()
                        csv_df_export.columns = [f"{str(col[0])}_{col[1]}" if isinstance(col, tuple) else str(col) for col in csv_df_export.columns.values]
                        csv_df_export = csv_df_export.reset_index()
                        try:
                            csv_str_export = csv_df_export.to_csv(index=False).encode('utf-8')
                            st.markdown("---"); st.download_button("Download Detailed Allocation (CSV)", csv_str_export, f"portfolio_allocation_{datetime.date.today()}.csv", 'text/csv', key='download-csv')
                        except Exception as e_csv_export_main: st.error(f"Error generating CSV: {e_csv_export_main}")

            except ValueError as e_val_run_main: st.error(f"Config/Allocation Error: {e_val_run_main}")
            except KeyError as e_key_run_main: st.error(f"Data Error (Missing Key): '{e_key_run_main}'. Trace: {traceback.format_exc()}")
            except Exception as e_gen_run_main: st.error(f"Unexpected error: {e_gen_run_main}"); st.error(f"Traceback: {traceback.format_exc()}")
        else: st.error("⚠️ Missing settings. Check sidebar (planning horizon, project selections).")

# --- Footer ---
st.markdown("---")
try:
    zurich_tz = pytz.timezone('Europe/Zurich')
    st.caption(f"Report generated: {datetime.datetime.now(zurich_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception: st.caption(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local)")
