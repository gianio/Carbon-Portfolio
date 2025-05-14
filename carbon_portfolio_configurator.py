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
import unicodedata # Added for the new data loading function

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
# Original Allocation Function
# ==================================
def allocate_portfolio(
    project_data: pd.DataFrame, selected_project_names: list, selected_years: list,
    start_year_portfolio: int, end_year_portfolio: int, constraint_type: str, annual_targets: dict,
    removal_target_percent_end_year: float, transition_speed: int, category_split: dict,
    favorite_project: str = None, priority_boost_percent: int = 10, # priority_boost_percent not used in original, but could be
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
            adj_candidates = projects_year_df_loop[(projects_year_df_loop['remaining_volume'] >= min_allocation_chunk) & (projects_year_df_loop[price_col_loop] > 0)].sort_values(by=['priority', price_col_loop], ascending=[False, True]).copy() # Original script uses 'priority' here, which is project-level.
            # For a more refined adjustment, could sort by 'final_priority' if it's consistently populated, or just price.
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
        is_overbudget_val = check_df_budget['Allocated Cost'] > check_df_budget['Target Budget'] * 1.001 # Allow for tiny float inaccuracies
        overbudget_df = check_df_budget[is_overbudget_val]
        if not overbudget_df.empty: st.warning(f"Budget slightly exceeded target cap in years: {overbudget_df['Year'].tolist()}. This can happen due to min_allocation_chunk effects.")
    return portfolio_details, yearly_summary_df_res

# ==================================
# Margin Calculation Functions
# ==================================
def get_margin_per_unit(project_data_row: pd.Series, allocated_price_this_year: float) -> float:
    if pd.isna(allocated_price_this_year) or allocated_price_this_year < 0: return 0.0
    base_price_col, thresh_price_col, mshare_col, fpp_col, pms_col = 'base price', 'threshold price', 'm_a', 'fixed purchase price', 'percental margin share'
    margin_pu = 0.0
    bp, tp, ms = project_data_row.get(base_price_col), project_data_row.get(thresh_price_col), project_data_row.get(mshare_col)
    try:
        if pd.notna(bp) and pd.notna(tp) and pd.notna(ms): margin_pu = max(0.1, allocated_price_this_year - float(tp)) * float(ms) + (float(tp) - float(bp))
        elif pd.notna(project_data_row.get(fpp_col)): margin_pu = allocated_price_this_year - float(project_data_row.get(fpp_col))
        elif pd.notna(project_data_row.get(pms_col)): margin_pu = allocated_price_this_year * float(project_data_row.get(pms_col))
    except: margin_pu = 0.0
    return margin_pu if pd.notna(margin_pu) and margin_pu != -np.inf and margin_pu != np.inf else 0.0

def add_margins_to_details_df(details_df: pd.DataFrame, project_master_data: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        expected_cols = list(details_df.columns)
        if 'margin_per_unit' not in expected_cols: expected_cols.append('margin_per_unit')
        if 'margin' not in expected_cols: expected_cols.append('margin')
        return pd.DataFrame(columns=expected_cols)

    rows_with_margins = []
    if 'project name' not in project_master_data.columns: # Defensive check
        st.error("Critical: 'project name' column missing in master data for margin calculation.")
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
            if isinstance(proj_data_row_margin, pd.DataFrame): proj_data_row_margin = proj_data_row_margin.iloc[0] # Handle duplicate project names if any (take first)
            margin_pu_val = get_margin_per_unit(proj_data_row_margin, price_used)
            margin_total = margin_pu_val * alloc_vol
        new_row_data.update({'margin_per_unit': margin_pu_val, 'margin': margin_total})
        rows_with_margins.append(new_row_data)

    if not rows_with_margins and not details_df.empty: # If loop didn't run but df was not empty
        temp_df = details_df.copy()
        temp_df['margin_per_unit'], temp_df['margin'] = 0.0, 0.0
        return temp_df
    return pd.DataFrame(rows_with_margins)

# ==================================
# NEW Allocation Function to Maximize Margin
# ==================================
def allocate_portfolio_maximize_margin(
    project_data: pd.DataFrame,
    selected_project_names: list,
    selected_years: list,
    constraint_type: str,  # 'Volume' or 'Budget'
    annual_targets: dict,  # {year: target_value}, acts as a cap
    get_margin_per_unit_func: callable, # Pass the actual margin function
    min_allocation_chunk: int = 1,
    min_target_fulfillment_percent: float = 0.0 # Range 0.0 to 1.0 (Currently only for potential future extension)
) -> tuple[dict, pd.DataFrame]:
    """
    Allocates portfolio to maximize total margin, subject to annual targets (as caps) and constraints.
    """
    portfolio_details = {year: [] for year in selected_years}
    yearly_summary_list = []
    # Note: 'Actual Removal Vol %' and 'Target Removal Vol %' are not calculated by this strategy.
    summary_cols = ['Year', f'Target {constraint_type}', 'Allocated Volume', 'Allocated Cost', 'Total Yearly Margin', 'Avg. Price', 'Avg. Margin per Unit']


    if not selected_project_names:
        st.warning("Margin Maximization: No projects selected.")
        return {}, pd.DataFrame(columns=summary_cols)

    project_data_selected = project_data[project_data['project name'].isin(selected_project_names)].copy()
    if project_data_selected.empty:
        st.warning("Margin Maximization: Selected projects not found in data.")
        return {}, pd.DataFrame(columns=summary_cols)

    required_base_cols = ['project name', 'project type'] # Add other cols needed by get_margin_per_unit_func if not passed via project_data_row
    for col in required_base_cols:
        if col not in project_data_selected.columns:
            raise ValueError(f"Margin Maximization: Input project_data missing required column for base processing: {col}")

    # Ensure margin-related columns exist, fill with NaN if not (get_margin_per_unit should handle NaNs)
    margin_calc_cols = ['base price', 'threshold price', 'm_a', 'fixed purchase price', 'percental margin share']
    for m_col in margin_calc_cols:
        if m_col not in project_data_selected.columns:
            project_data_selected[m_col] = np.nan # Or some other sensible default if get_margin_per_unit expects it

    for year_check in selected_years:
        price_col_check = f"price {year_check}"
        volume_col_check = f"available volume {year_check}"
        
        # Ensure numeric types for the current year's price and volume if columns exist
        if price_col_check in project_data_selected.columns:
            project_data_selected[price_col_check] = pd.to_numeric(project_data_selected[price_col_check], errors='coerce').fillna(0.0).clip(lower=0.0)
        else: # If price column for a year is entirely missing, we can't proceed for that year
            project_data_selected[price_col_check] = 0.0 # Add it as zero so rows are filtered out later if used

        if volume_col_check in project_data_selected.columns:
            project_data_selected[volume_col_check] = pd.to_numeric(project_data_selected[volume_col_check], errors='coerce').fillna(0).astype(int).clip(lower=0)
        else: # If volume column for a year is entirely missing
            project_data_selected[volume_col_check] = 0 # Add it as zero


    for year in selected_years:
        yearly_target_cap = annual_targets.get(year, 0)
        price_col = f"price {year}"
        volume_col = f"available volume {year}"

        year_allocations_list = []
        current_year_total_volume = 0
        current_year_total_cost = 0.0
        current_year_total_margin = 0.0

        def create_empty_year_summary_for_margin_strat():
            return {
                'Year': year, f'Target {constraint_type}': yearly_target_cap,
                'Allocated Volume': 0, 'Allocated Cost': 0.0, 'Total Yearly Margin': 0.0,
                'Avg. Price': 0.0, 'Avg. Margin per Unit': 0.0
            }

        if yearly_target_cap <= 0:
            yearly_summary_list.append(create_empty_year_summary_for_margin_strat())
            portfolio_details[year] = []
            continue

        # Filter projects that have valid data for the current year
        # Ensure all columns needed by get_margin_per_unit_func are selected, plus year-specific ones
        cols_to_select_for_year = list(set(required_base_cols + margin_calc_cols + [price_col, volume_col, 'project name']))
        # Filter out any columns that might not exist in project_data_selected (e.g. if a year column was missing)
        cols_to_select_for_year = [c for c in cols_to_select_for_year if c in project_data_selected.columns]

        projects_for_year_df = project_data_selected[cols_to_select_for_year].copy()
        
        projects_for_year_df = projects_for_year_df[
            (project_data_selected[price_col] > 0) &
            (project_data_selected[volume_col] >= min_allocation_chunk)
        ].copy() # Re-copy after filtering to avoid SettingWithCopyWarning

        if projects_for_year_df.empty:
            yearly_summary_list.append(create_empty_year_summary_for_margin_strat())
            portfolio_details[year] = []
            continue

        projects_for_year_df['margin_per_unit'] = projects_for_year_df.apply(
            lambda row: get_margin_per_unit_func(row, row[price_col]), axis=1
        )
        
        profitable_projects_df = projects_for_year_df[projects_for_year_df['margin_per_unit'] > 1e-6].copy() # Consider only positive margin

        if profitable_projects_df.empty:
            yearly_summary_list.append(create_empty_year_summary_for_margin_strat())
            portfolio_details[year] = []
            continue

        if constraint_type == 'Budget':
            profitable_projects_df['rank_metric'] = profitable_projects_df.apply(
                lambda row: row['margin_per_unit'] / row[price_col] if row[price_col] > 1e-6 else -float('inf'),
                axis=1
            )
        else:  # Volume constraint
            profitable_projects_df['rank_metric'] = profitable_projects_df['margin_per_unit']
        
        sorted_profitable_projects = profitable_projects_df.sort_values(
            by=['rank_metric', 'project name'], ascending=[False, True] # Highest margin/rank first
        )

        project_volume_tracker_this_year = sorted_profitable_projects.set_index('project name')[volume_col].to_dict()

        for _, project_row in sorted_profitable_projects.iterrows():
            project_name = project_row['project name']
            project_price = project_row[price_col]
            project_margin_pu = project_row['margin_per_unit']
            project_type = project_row['project type'] # Ensure 'project type' is in projects_for_year_df
            
            current_project_available_vol = project_volume_tracker_this_year.get(project_name, 0)

            if current_project_available_vol < min_allocation_chunk:
                continue

            volume_to_allocate = 0
            if constraint_type == 'Volume':
                remaining_target_metric = yearly_target_cap - current_year_total_volume
                if remaining_target_metric < min_allocation_chunk : break 
                
                max_vol_by_target = remaining_target_metric
                max_vol_possible = min(current_project_available_vol, max_vol_by_target)
                volume_to_allocate = math.floor(max_vol_possible / min_allocation_chunk) * min_allocation_chunk

            elif constraint_type == 'Budget':
                remaining_target_metric = yearly_target_cap - current_year_total_cost
                if remaining_target_metric < (project_price * min_allocation_chunk * 0.99) and remaining_target_metric < 0.01 : break # Check if enough budget for at least one chunk or very small amount

                max_vol_by_budget = remaining_target_metric / project_price if project_price > 1e-6 else 0
                max_vol_possible = min(current_project_available_vol, max_vol_by_budget)
                volume_to_allocate = math.floor(max_vol_possible / min_allocation_chunk) * min_allocation_chunk
            
            if volume_to_allocate >= min_allocation_chunk:
                cost_of_allocation = volume_to_allocate * project_price
                margin_from_allocation = volume_to_allocate * project_margin_pu

                if constraint_type == 'Budget' and (current_year_total_cost + cost_of_allocation > yearly_target_cap * 1.0001):
                    if current_year_total_cost < yearly_target_cap :
                        budget_left = yearly_target_cap - current_year_total_cost
                        cost_per_chunk = min_allocation_chunk * project_price
                        if cost_per_chunk > 1e-6: # Avoid division by zero if price is effectively zero
                            num_chunks_can_fit = math.floor(budget_left / cost_per_chunk)
                            if num_chunks_can_fit > 0:
                                volume_to_allocate = num_chunks_can_fit * min_allocation_chunk
                                cost_of_allocation = volume_to_allocate * project_price
                                margin_from_allocation = volume_to_allocate * project_margin_pu
                            else: volume_to_allocate = 0 
                        else: volume_to_allocate = 0
                    else: volume_to_allocate = 0

                if volume_to_allocate >= min_allocation_chunk:
                    year_allocations_list.append({
                        'project name': project_name,
                        'type': project_type,
                        'allocated_volume': volume_to_allocate,
                        'allocated_cost': cost_of_allocation,
                        'allocated_margin': margin_from_allocation, # New field
                        'price_used': project_price,
                        'margin_per_unit': project_margin_pu # New field
                    })
                    current_year_total_volume += volume_to_allocate
                    current_year_total_cost += cost_of_allocation
                    current_year_total_margin += margin_from_allocation
                    project_volume_tracker_this_year[project_name] -= volume_to_allocate
        
        portfolio_details[year] = year_allocations_list
        
        avg_price_calc = (current_year_total_cost / current_year_total_volume) if current_year_total_volume > 0 else 0.0
        avg_margin_pu_calc = (current_year_total_margin / current_year_total_volume) if current_year_total_volume > 0 else 0.0

        yearly_summary_list.append({
            'Year': year,
            f'Target {constraint_type}': yearly_target_cap,
            'Allocated Volume': current_year_total_volume,
            'Allocated Cost': current_year_total_cost,
            'Total Yearly Margin': current_year_total_margin,
            'Avg. Price': avg_price_calc,
            'Avg. Margin per Unit': avg_margin_pu_calc # New field in summary
        })

    yearly_summary_df_final = pd.DataFrame(yearly_summary_list)
    if yearly_summary_df_final.empty:
        yearly_summary_df_final = pd.DataFrame(columns=summary_cols)
    else:
        yearly_summary_df_final = yearly_summary_df_final[summary_cols]

    return portfolio_details, yearly_summary_df_final


# ==================================
# Streamlit App Layout & Logic
# ==================================
with st.sidebar:
    st.markdown("## 1. Load Data")
    df_upload = st.file_uploader("Upload Project Data CSV", type="csv", key="uploader_sidebar", help="Required: project name, project type, priority, price_YYYY, available_volume_YYYY. Optional margins: base price, threshold price, m_a, fixed purchase price, percental margin share. Optional: description, project_link.")
    
    # Initialize session state keys
    default_values = {
        'working_data_full': None, 'selected_years': [], 'selected_projects': [], 
        'project_names': [], 'favorite_projects_selection': [], 
        'actual_start_year': None, 'actual_end_year': None, 
        'available_years_in_data': [], 'constraint_type': 'Volume', 
        'removal_target_end_year': 0.8, 'transition_speed': 5, 
        'category_split': {'technical removal': 0.5, 'natural removal': 0.5}, 
        'annual_targets': {}, 'master_target': None, 'data_loaded_successfully': False, 
        'years_slider_sidebar': 5, 'min_fulfillment_perc': 95, 
        'removal_preference_slider': 5, 'min_alloc_chunk': 1,
        'allocation_strategy': 'Priority-Based Allocation' # New session state for strategy
    }
    for key, default_value in default_values.items():
        if key not in st.session_state: st.session_state[key] = default_value

    if df_upload:
        @st.cache_data
        def load_and_prepare_data_simplified(uploaded_file):
            try:
                data = pd.read_csv(uploaded_file, encoding='utf-8-sig', skipinitialspace=True)
                standardized_columns = []
                for col in data.columns:
                    col_str = str(col)
                    normalized_col = unicodedata.normalize('NFKC', col_str)
                    lower_col = normalized_col.lower()
                    stripped_col = lower_col.strip()
                    final_col = stripped_col.replace(' ', '_')
                    standardized_columns.append(final_col)
                data.columns = standardized_columns
            except Exception as read_error:
                return None, f"Error reading or initially processing CSV file: {read_error}. Ensure it's a valid CSV (comma-separated by default).", [], [], []

            core_cols_std = ['project_name', 'project_type', 'priority']
            optional_cols_std = ['description', 'project_link']
            margin_cols_std = ['base_price', 'threshold_price', 'm_a', 'fixed_purchase_price', 'percental_margin_share']
            data_columns_set = set(data.columns)

            missing_essential = [col for col in core_cols_std if col not in data_columns_set]
            if missing_essential:
                err_msg = (
                    f"CSV missing essential columns after standardization. "
                    f"Expected: {core_cols_std}. Found in file: {list(data.columns)}. Missing: {missing_essential}."
                )
                return None, err_msg, [], [], []

            if 'treshold_price' in data.columns and 'threshold_price' not in data.columns:
                data.rename(columns={'treshold_price': 'threshold_price'}, inplace=True)

            for m_col in margin_cols_std:
                if m_col not in data.columns: data[m_col] = np.nan

            available_years = set()
            year_data_cols_found_std = []
            numeric_prefixes_std = ['price_', 'available_volume_']
            project_names_list_intermediate = []
            if 'project_name' in data.columns:
                try: project_names_list_intermediate = sorted(data['project_name'].astype(str).unique().tolist())
                except Exception: project_names_list_intermediate = []

            for col_std in data.columns:
                for prefix in numeric_prefixes_std:
                    if col_std.startswith(prefix):
                        year_part = col_std[len(prefix):]
                        if year_part.isdigit() and len(year_part) == 4:
                            try:
                                year_val = int(year_part)
                                if 2000 <= year_val <= 2100:
                                    available_years.add(year_val)
                                    year_data_cols_found_std.append(col_std)
                            except ValueError: pass
                        break
            
            if not available_years:
                return None, "No valid year columns (e.g., 'price_YYYY', 'available_volume_YYYY') detected.", [], project_names_list_intermediate, []

            available_years_list = sorted(list(available_years))
            cols_to_convert_numeric = ['priority'] + margin_cols_std + year_data_cols_found_std
            for col_num_conv in list(set(cols_to_convert_numeric)):
                if col_num_conv in data.columns:
                    data[col_num_conv] = pd.to_numeric(data[col_num_conv], errors='coerce')

            data['priority'] = data['priority'].fillna(0).clip(lower=0)
            for m_col_proc in margin_cols_std:
                if m_col_proc in data.columns:
                    if m_col_proc in ['base_price', 'threshold_price', 'fixed_purchase_price']:
                        data[m_col_proc] = data[m_col_proc].apply(lambda x: x if pd.notna(x) and x >= 0 else np.nan)

            for col_yr_proc in year_data_cols_found_std:
                if col_yr_proc.startswith('available_volume_'):
                    data[col_yr_proc] = data[col_yr_proc].fillna(0).apply(lambda x: max(0, int(x)) if pd.notna(x) else 0).clip(lower=0)
                elif col_yr_proc.startswith('price_'):
                    data[col_yr_proc] = data[col_yr_proc].fillna(0.0).apply(lambda x: max(0.0, float(x)) if pd.notna(x) else 0.0).clip(lower=0)

            invalid_types_list = []
            if 'project_type' in data.columns:
                data['project_type'] = data['project_type'].astype(str).str.lower().str.strip()
                valid_types = ['reduction', 'technical removal', 'natural removal']
                invalid_rows_mask = ~data['project_type'].isin(valid_types)
                if invalid_rows_mask.any():
                    invalid_types_list = data.loc[invalid_rows_mask, 'project_type'].unique().tolist()
                    data = data[~invalid_rows_mask].copy()
                if data.empty:
                    return None, f"All rows had invalid project types. Valid: {valid_types}. Found: {invalid_types_list}", available_years_list, [], invalid_types_list

            all_cols_to_potentially_keep = list(set(core_cols_std + optional_cols_std + margin_cols_std + year_data_cols_found_std))
            final_cols_to_keep = [col for col in all_cols_to_potentially_keep if col in data.columns]
            data = data[final_cols_to_keep].copy()

            final_rename_map_dict = {
                'project_name': 'project name', 'project_type': 'project type', 'priority': 'priority',
                'description': 'Description', 'project_link': 'Project Link',
                'base_price': 'base price', 'threshold_price': 'threshold price', 'm_a': 'm_a',
                'fixed_purchase_price': 'fixed purchase price', 'percental_margin_share': 'percental margin share'
            }
            for yr_col_map in year_data_cols_found_std:
                if yr_col_map in data.columns:
                    final_rename_map_dict[yr_col_map] = yr_col_map.replace('_', ' ')
            
            actual_rename_map = {k: v for k, v in final_rename_map_dict.items() if k in data.columns}
            data.rename(columns=actual_rename_map, inplace=True)
            
            project_names_output = sorted(data['project name'].unique().tolist()) if 'project name' in data.columns else []
            return data, None, available_years_list, project_names_output, invalid_types_list

        try:
            data_main, err_msg, years_data, proj_names, invalid_types = load_and_prepare_data_simplified(df_upload)
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
            
            # Allocation Strategy Selector
            st.session_state.allocation_strategy = st.radio(
                "Allocation Strategy:",
                ('Priority-Based Allocation', 'Maximize Margin Allocation'),
                index=0 if st.session_state.get('allocation_strategy', 'Priority-Based Allocation') == 'Priority-Based Allocation' else 1,
                key='allocation_strategy_selector',
                horizontal=True,
                help="Choose 'Priority-Based' for standard allocation or 'Maximize Margin' to prioritize projects with the highest potential margin (annual targets act as caps)."
            )
            st.markdown("---") # Visual separator

            min_yr_sb, max_yr_sb = min(years_ui), max(years_ui)
            max_yrs_plan_sb = max(1, max_yr_sb - min_yr_sb + 1)
            curr_yrs_plan_val_sb = st.session_state.get('years_slider_sidebar', 5);
            try: curr_yrs_plan_val_sb = int(curr_yrs_plan_val_sb)
            except: curr_yrs_plan_val_sb = 5
            curr_yrs_plan_val_sb = max(1, min(curr_yrs_plan_val_sb, max_yrs_plan_sb))
            yrs_to_plan_sb = st.number_input(f"Years to Plan (Starting {min_yr_sb})", 1, max_yrs_plan_sb, curr_yrs_plan_val_sb, 1, key='years_slider_sidebar_widget', help=f"Enter # years (1 to {max_yrs_plan_sb}).")
            st.session_state.years_slider_sidebar = yrs_to_plan_sb
            start_yr_sel_sb, end_yr_sel_sb = min_yr_sb, min_yr_sb + yrs_to_plan_sb - 1
            
            # Ensure selected years have data
            valid_years_with_data = []
            for yr_potential in range(start_yr_sel_sb, end_yr_sel_sb + 1):
                if f"price {yr_potential}" in data_ui.columns and f"available volume {yr_potential}" in data_ui.columns:
                     # Check if at least one project has non-zero price and volume for this year
                    if not data_ui[(data_ui[f"price {yr_potential}"] > 0) & (data_ui[f"available volume {yr_potential}"] > 0)].empty:
                        valid_years_with_data.append(yr_potential)

            st.session_state.selected_years = valid_years_with_data

            if not st.session_state.selected_years:
                st.sidebar.error(f"No valid data for planning period ({start_yr_sel_sb}-{end_yr_sel_sb}). Adjust 'Years to Plan' or check data integrity (prices/volumes > 0).")
                st.session_state.actual_start_year, st.session_state.actual_end_year = None, None
            else:
                st.session_state.actual_start_year, st.session_state.actual_end_year = min(st.session_state.selected_years), max(st.session_state.selected_years)
                st.sidebar.markdown(f"Planning Horizon: **{st.session_state.actual_start_year} - {st.session_state.actual_end_year}**")
                st.session_state.constraint_type = st.radio("Constraint Type:", ('Volume', 'Budget'), index=['Volume', 'Budget'].index(st.session_state.get('constraint_type', 'Volume')), key='constraint_type_sidebar_widget', horizontal=True)
                
                target_label_suffix = " (Target/Cap)" if st.session_state.allocation_strategy == 'Maximize Margin Allocation' else " (Target)"
                st.markdown(f"### Annual Target Settings{target_label_suffix}")


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
                # Min fulfillment is more relevant for priority-based. For margin-max, target is a cap.
                disable_min_fulfill = st.session_state.allocation_strategy == 'Maximize Margin Allocation'
                st.session_state.min_fulfillment_perc = st.sidebar.slider(
                    f"Min. Target Fulfillment (%) {'(Priority-Based Only)' if disable_min_fulfill else ''}", 
                    50, 100, st.session_state.get('min_fulfillment_perc', 95), 
                    key='min_fulfill_perc_sidebar',
                    disabled=disable_min_fulfill
                )
                st.session_state.min_alloc_chunk = int(st.sidebar.number_input("Min. Allocation Unit (t)", 1, step=1, value=st.session_state.get('min_alloc_chunk', 1), key='min_alloc_chunk_sidebar') or 1)

                # These settings are specific to Priority-Based Allocation
                is_priority_strategy = st.session_state.allocation_strategy == 'Priority-Based Allocation'
                st.sidebar.markdown(f"### Removal Volume Transition {'(Priority-Based Only)' if not is_priority_strategy else ''}")
                reduc_present_sb_val = 'reduction' in data_ui['project type'].unique() if 'project type' in data_ui.columns else False
                st.sidebar.info("Applies if 'Reduction' projects selected (Priority-Based)." if reduc_present_sb_val and is_priority_strategy else "Inactive or N/A for current strategy.")
                end_yr_help_sb_val = st.session_state.actual_end_year or "end year"
                try: rem_target_default_sb_val = int(float(st.session_state.get('removal_target_end_year', 0.8)) * 100)
                except: rem_target_default_sb_val = 80
                st.session_state.removal_target_end_year = st.sidebar.slider(f"Target Removal Vol % ({end_yr_help_sb_val})", 0, 100, rem_target_default_sb_val, key='removal_perc_slider_sidebar', disabled=not (reduc_present_sb_val and is_priority_strategy)) / 100.0
                st.session_state.transition_speed = st.sidebar.slider("Transition Speed", 1, 10, st.session_state.get('transition_speed', 5), key='transition_speed_slider_sidebar', disabled=not (reduc_present_sb_val and is_priority_strategy))
                
                st.sidebar.markdown(f"### Removal Category Preference {'(Priority-Based Only)' if not is_priority_strategy else ''}")
                rem_types_present_sb_val = any(pt in data_ui['project type'].unique() for pt in ['technical removal', 'natural removal']) if 'project type' in data_ui.columns else False
                rem_pref_val_sb_val = st.sidebar.slider("Technical vs Natural Preference", 1, 10, st.session_state.get('removal_preference_slider', 5), format="%d", key='removal_pref_slider_sidebar', disabled=not (rem_types_present_sb_val and is_priority_strategy))
                st.session_state['removal_preference_slider'] = rem_pref_val_sb_val; tech_pref_ratio_sb_val = (rem_pref_val_sb_val - 1) / 9.0
                st.session_state.category_split = {'technical removal': tech_pref_ratio_sb_val, 'natural removal': 1.0 - tech_pref_ratio_sb_val}
                
                st.sidebar.markdown("## 3. Select Projects")
                if not names_ui: st.sidebar.warning("No projects available.")
                else:
                    st.session_state.selected_projects = st.sidebar.multiselect("Select projects to include:", options=names_ui, default=st.session_state.get('selected_projects', names_ui), key='project_selector_sidebar')
                    if 'priority' in data_ui.columns and is_priority_strategy: # Favorite project boost only for priority-based
                        boost_opts_sb_val = [p for p in names_ui if p in st.session_state.selected_projects]
                        if boost_opts_sb_val:
                            fav_default_sb_val = [f for f in st.session_state.get('favorite_projects_selection', []) if f in boost_opts_sb_val][:1]
                            st.session_state.favorite_projects_selection = st.sidebar.multiselect("Favorite Project (Priority Boost - Priority-Based Only):", options=boost_opts_sb_val, default=fav_default_sb_val, key='favorite_selector_sidebar', max_selections=1)
                        else: st.session_state.favorite_projects_selection = []
                    else: st.session_state.favorite_projects_selection = []

# ==================================
# Main Page Content
# ==================================
st.markdown(f"<h1 style='color: #8ca734;'>Carbon Portfolio Builder</h1>", unsafe_allow_html=True)
st.markdown(f"Current Allocation Strategy: **{st.session_state.get('allocation_strategy', 'N/A')}**")
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
    elif not st.session_state.get('selected_years'): st.warning("⚠️ No valid years for planning. Adjust 'Years to Plan' (Section 2) or check data (prices/volumes > 0 for selected range).")
    else:
        # Check required keys based on strategy
        allocation_strategy = st.session_state.allocation_strategy
        
        base_required_keys = ['working_data_full', 'selected_projects', 'selected_years', 
                              'actual_start_year', 'actual_end_year', 'constraint_type', 
                              'annual_targets', 'min_alloc_chunk']
        
        priority_strategy_keys = ['removal_target_end_year', 'transition_speed', 
                                  'category_split', 'favorite_projects_selection', 
                                  'min_fulfillment_perc']
        
        required_keys_main = base_required_keys
        if allocation_strategy == 'Priority-Based Allocation':
            required_keys_main.extend(priority_strategy_keys)

        keys_ok = all(st.session_state.get(k) is not None for k in required_keys_main 
                      if k not in ['annual_targets', 'favorite_projects_selection', 'category_split']) and \
                    isinstance(st.session_state.get('annual_targets'), dict) and \
                    isinstance(st.session_state.get('favorite_projects_selection', []), list) and \
                    isinstance(st.session_state.get('category_split', {}), dict)


        if keys_ok:
            try:
                results_run = {}
                summary_df_run = pd.DataFrame()
                
                fav_proj_run = st.session_state.favorite_projects_selection[0] if st.session_state.favorite_projects_selection else None
                annual_targets_run = st.session_state.annual_targets
                if not annual_targets_run and st.session_state.selected_years: # Ensure targets dict is not empty if years are selected
                    annual_targets_run = {yr: st.session_state.master_target for yr in st.session_state.selected_years}


                with st.spinner(f"Calculating portfolio using {allocation_strategy}..."):
                    if allocation_strategy == 'Priority-Based Allocation':
                        if st.session_state.constraint_type == 'Budget': st.info(f"**Budget Mode (Priority-Based):** Projects get budget via weighted priority & price...")
                        st.success(f"**Allocation Goal (Priority-Based):** Attempting ≥ **{st.session_state.min_fulfillment_perc}%** of annual target {st.session_state.constraint_type}.")
                        results_run, summary_df_run = allocate_portfolio(
                            project_data=st.session_state.working_data_full, 
                            selected_project_names=st.session_state.selected_projects,
                            selected_years=st.session_state.selected_years, 
                            start_year_portfolio=st.session_state.actual_start_year,
                            end_year_portfolio=st.session_state.actual_end_year, 
                            constraint_type=st.session_state.constraint_type,
                            annual_targets=annual_targets_run, 
                            removal_target_percent_end_year=st.session_state.removal_target_end_year,
                            transition_speed=st.session_state.transition_speed, 
                            category_split=st.session_state.category_split,
                            favorite_project=fav_proj_run, 
                            min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0,
                            min_allocation_chunk=st.session_state.min_alloc_chunk
                        )
                    elif allocation_strategy == 'Maximize Margin Allocation':
                        st.info(f"**Maximize Margin Mode:** Allocating projects to maximize margin, up to annual target {st.session_state.constraint_type} (as a cap).")
                        # min_fulfillment_perc is not directly used by this strategy's core logic but passed for signature consistency
                        results_run, summary_df_run = allocate_portfolio_maximize_margin(
                            project_data=st.session_state.working_data_full,
                            selected_project_names=st.session_state.selected_projects,
                            selected_years=st.session_state.selected_years,
                            constraint_type=st.session_state.constraint_type,
                            annual_targets=annual_targets_run,
                            get_margin_per_unit_func=get_margin_per_unit, # Pass the actual function
                            min_allocation_chunk=st.session_state.min_alloc_chunk,
                            min_target_fulfillment_percent=st.session_state.min_fulfillment_perc / 100.0 
                        )

                details_list_run = []
                details_df_with_margins_run = pd.DataFrame()

                if results_run:
                    if allocation_strategy == 'Priority-Based Allocation':
                        for yr_res, projs_res in results_run.items():
                            if projs_res:
                                for proj_detail in projs_res:
                                    if isinstance(proj_detail, dict) and (proj_detail.get('allocated_volume', 0) >= st.session_state.min_alloc_chunk or proj_detail.get('allocated_cost', 0) > 1e-6):
                                        details_list_run.append({
                                            'year': yr_res, 
                                            'project name': proj_detail.get('project name'), 
                                            'type': proj_detail.get('type'), 
                                            'volume': proj_detail.get('allocated_volume', 0), 
                                            'price': proj_detail.get('price_used', None), 
                                            'cost': proj_detail.get('allocated_cost', 0.0)
                                        })
                        temp_details_df = pd.DataFrame(details_list_run)
                        if not temp_details_df.empty:
                             details_df_with_margins_run = add_margins_to_details_df(temp_details_df.copy(), st.session_state.working_data_full)
                        else: # Ensure columns if empty
                            details_df_with_margins_run = pd.DataFrame(columns=['year', 'project name', 'type', 'volume', 'price', 'cost', 'margin_per_unit', 'margin'])


                    elif allocation_strategy == 'Maximize Margin Allocation':
                        for yr_res, projs_res in results_run.items():
                            if projs_res:
                                for proj_detail in projs_res: # These dicts already contain margin info
                                    if isinstance(proj_detail, dict) and (proj_detail.get('allocated_volume', 0) >= st.session_state.min_alloc_chunk or proj_detail.get('allocated_cost', 0) > 1e-6):
                                        details_list_run.append({
                                            'year': yr_res,
                                            'project name': proj_detail.get('project name'),
                                            'type': proj_detail.get('type'),
                                            'volume': proj_detail.get('allocated_volume', 0),
                                            'price': proj_detail.get('price_used', None),
                                            'cost': proj_detail.get('allocated_cost', 0.0),
                                            'margin_per_unit': proj_detail.get('margin_per_unit', 0.0),
                                            'margin': proj_detail.get('allocated_margin', 0.0)
                                        })
                        details_df_with_margins_run = pd.DataFrame(details_list_run)
                        if details_df_with_margins_run.empty: # Ensure columns if empty
                             details_df_with_margins_run = pd.DataFrame(columns=['year', 'project name', 'type', 'volume', 'price', 'cost', 'margin_per_unit', 'margin'])


                total_portfolio_margin_run = 0.0
                if not details_df_with_margins_run.empty and 'margin' in details_df_with_margins_run.columns:
                    total_portfolio_margin_run = details_df_with_margins_run['margin'].sum()

                # Update summary_df_run with calculated total yearly margins if not already there (e.g. from priority-based)
                # For margin-max strategy, 'Total Yearly Margin' is already in summary_df_run
                if allocation_strategy == 'Priority-Based Allocation':
                    if 'Total Yearly Margin' not in summary_df_run.columns: summary_df_run['Total Yearly Margin'] = 0.0 # Should exist from func
                    if not summary_df_run.empty and not details_df_with_margins_run.empty and \
                       'year' in details_df_with_margins_run.columns and 'margin' in details_df_with_margins_run.columns and \
                       not details_df_with_margins_run.groupby('year')['margin'].sum().empty:
                        actual_yearly_margins_series = details_df_with_margins_run.groupby('year')['margin'].sum()
                        summary_df_run['Total Yearly Margin'] = summary_df_run['Year'].map(actual_yearly_margins_series).fillna(0.0)
                    elif not summary_df_run.empty : summary_df_run['Total Yearly Margin'] = 0.0


                st.markdown("## Portfolio Summary"); colL, colM, colR = st.columns([1.5, 1.5, 1.2], gap="large")
                with colL:
                    st.markdown("#### Key Metrics (Overall)")
                    tot_cost, tot_vol, avg_price_overall = 0.0, 0, 0.0
                    if not summary_df_run.empty:
                        tot_cost = summary_df_run['Allocated Cost'].sum() if 'Allocated Cost' in summary_df_run else 0.0
                        tot_vol = summary_df_run['Allocated Volume'].sum() if 'Allocated Volume' in summary_df_run else 0
                        avg_price_overall = tot_cost / tot_vol if tot_vol > 0 else 0.0
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Cost</b> € {tot_cost:,.2f}</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Volume</b> {tot_vol:,.0f} t</div>""", unsafe_allow_html=True)
                with colM:
                    st.markdown("#### &nbsp;")
                    st.markdown(f"""<div class="metric-box"><b>Total Portfolio Margin</b> € {total_portfolio_margin_run:,.2f}</div>""", unsafe_allow_html=True)
                    st.markdown(f"""<div class="metric-box"><b>Overall Average Price</b> € {avg_price_overall:,.2f} /t</div>""", unsafe_allow_html=True)
                with colR:
                    st.markdown("#### Volume by Project Type")
                    df_for_pie = details_df_with_margins_run
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
                df_for_plots_tables = details_df_with_margins_run # Already has margin info
                if 'margin' not in df_for_plots_tables.columns and not df_for_plots_tables.empty: # Fallback if margin somehow missing
                    df_for_plots_tables['margin'] = 0.0
                if 'margin_per_unit' not in df_for_plots_tables.columns and not df_for_plots_tables.empty:
                    df_for_plots_tables['margin_per_unit'] = 0.0


                if df_for_plots_tables.empty : st.warning("No detailed project allocations for plots/tables.")
                else:
                    st.markdown("### Portfolio Composition & Price Over Time")
                    if 'year' in df_for_plots_tables.columns: df_for_plots_tables['year'] = df_for_plots_tables['year'].astype(int)
                    
                    # Ensure all expected agg columns exist before groupby
                    agg_dict = {}
                    if 'volume' in df_for_plots_tables.columns: agg_dict['volume'] = ('volume', 'sum')
                    if 'cost' in df_for_plots_tables.columns: agg_dict['cost'] = ('cost', 'sum')
                    if 'margin' in df_for_plots_tables.columns: agg_dict['margin'] = ('margin', 'sum')
                    
                    if not agg_dict: # No data to aggregate
                         summary_plot_data_run = pd.DataFrame(columns=['year', 'type'])
                    else:
                        summary_plot_data_run = df_for_plots_tables.groupby(['year', 'type'], as_index=False).agg(**agg_dict)

                    if 'margin' not in summary_plot_data_run.columns and not summary_plot_data_run.empty: summary_plot_data_run['margin'] = 0.0 # Ensure margin column

                    price_summary_data_run = summary_df_run[['Year', 'Avg. Price']].rename(columns={'Year':'year', 'Avg. Price':'avg_price'}) if not summary_df_run.empty and 'Avg. Price' in summary_df_run.columns else pd.DataFrame(columns=['year', 'avg_price'])
                    fig_comp_run = make_subplots(specs=[[{"secondary_y": True}]])
                    y_metric_plot = 'volume' if st.session_state.constraint_type == 'Volume' else 'cost'
                    y_label_plot = f"Allocated {y_metric_plot.capitalize()} ({'t' if y_metric_plot == 'volume' else '€'})"
                    y_hover_plot_template = f"{y_metric_plot.capitalize()}: %{{y:{'{:,.0f}' if y_metric_plot == 'volume' else '€{:,.2f}'}}}<extra></extra>"
                    
                    for t_name_plot_loop in ['reduction', 'natural removal', 'technical removal']:
                        if not summary_plot_data_run.empty and 'type' in summary_plot_data_run.columns and t_name_plot_loop in summary_plot_data_run['type'].unique():
                            df_type_plot_loop = summary_plot_data_run[summary_plot_data_run['type'] == t_name_plot_loop]
                            if not df_type_plot_loop.empty and y_metric_plot in df_type_plot_loop and df_type_plot_loop[y_metric_plot].sum() > 1e-6 :
                                fig_comp_run.add_trace(go.Bar(x=df_type_plot_loop['year'], y=df_type_plot_loop[y_metric_plot], name=t_name_plot_loop.replace('_', ' ').capitalize(), marker_color=type_color_map.get(t_name_plot_loop, default_color), hovertemplate=f'Year: %{{x}}<br>Type: {t_name_plot_loop.replace("_", " ").capitalize()}<br>{y_hover_plot_template}'), secondary_y=False)
                    
                    if not price_summary_data_run.empty: fig_comp_run.add_trace(go.Scatter(x=price_summary_data_run['year'], y=price_summary_data_run['avg_price'], name='Avg Price (€/t)', mode='lines+markers', marker_symbol='circle', marker_size=8, line={"color":'#1B5E20', "width":3}, hovertemplate='Year: %{x}<br>Avg Price: €%{y:,.2f}/t<extra></extra>'), secondary_y=True)
                    
                    # Conditional trace for Actual Removal Vol %
                    if not summary_df_run.empty and 'Actual Removal Vol %' in summary_df_run.columns: 
                        fig_comp_run.add_trace(go.Scatter(x=summary_df_run['Year'], y=summary_df_run['Actual Removal Vol %'], name='Actual Removal Vol %', mode='lines+markers', line=dict(color='darkorange', dash='dash'), marker_symbol='star', marker_size=8, hovertemplate='Year: %{x}<br>Actual Removal: %{y:.1f}%<extra></extra>'), secondary_y=True)
                    
                    y2_max_plot_val = 105.0
                    if not price_summary_data_run.empty and 'avg_price' in price_summary_data_run.columns and price_summary_data_run['avg_price'].notna().any(): y2_max_plot_val = max(y2_max_plot_val, price_summary_data_run['avg_price'].max() * 1.1 if pd.notna(price_summary_data_run['avg_price'].max()) else y2_max_plot_val)
                    if not summary_df_run.empty and 'Actual Removal Vol %' in summary_df_run.columns and summary_df_run['Actual Removal Vol %'].notna().any(): y2_max_plot_val = max(y2_max_plot_val, summary_df_run['Actual Removal Vol %'].max() * 1.1 if pd.notna(summary_df_run['Actual Removal Vol %'].max()) else y2_max_plot_val)
                    
                    fig_comp_run.update_layout(xaxis_title='Year', yaxis_title=y_label_plot, yaxis2_title='Avg Price (€/t) / Rem. %', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", margin=dict(t=20, l=0, r=0, b=0), yaxis_rangemode='tozero', yaxis2=dict(rangemode='tozero', range=[0, y2_max_plot_val]), hovermode="x unified")
                    if st.session_state.selected_years: fig_comp_run.update_xaxes(tickmode='array', tickvals=st.session_state.selected_years, dtick=1)
                    st.plotly_chart(fig_comp_run, use_container_width=True)

                    st.markdown("### Detailed Allocation by Project and Year")
                    pivot_display_run = pd.DataFrame()
                    df_for_pivot_main = df_for_plots_tables # This now consistently has 'margin'
                    
                    if not df_for_pivot_main.empty and 'year' in df_for_pivot_main.columns:
                        try:
                            df_for_pivot_main['year'] = pd.to_numeric(df_for_pivot_main['year'])
                            
                            # Ensure all pivot values are present
                            pivot_values_list = []
                            if 'volume' in df_for_pivot_main: pivot_values_list.append('volume')
                            if 'cost' in df_for_pivot_main: pivot_values_list.append('cost')
                            if 'price' in df_for_pivot_main: pivot_values_list.append('price')
                            if 'margin' in df_for_pivot_main: pivot_values_list.append('margin')

                            if not pivot_values_list: # No values to pivot
                                raise ValueError("No data columns (volume, cost, price, margin) available for pivot table.")

                            pivot_intermediate_run = pd.pivot_table(df_for_pivot_main, 
                                                                    values=pivot_values_list, 
                                                                    index=['project name', 'type'], 
                                                                    columns='year', 
                                                                    aggfunc={'volume': 'sum', 'cost': 'sum', 'price': 'first', 'margin': 'sum'})

                            pivot_final_run = pd.DataFrame()
                            yrs_pivot_run_list = []
                            if not pivot_intermediate_run.empty:
                                pivot_final_run = pivot_intermediate_run.swaplevel(0, 1, axis=1)
                                metric_order_run = ['volume', 'cost', 'price', 'margin'] # Desired order
                                # Filter metric_order_run to only include metrics actually present in pivot_final_run.columns.get_level_values(1)
                                actual_metrics_in_pivot = list(pivot_final_run.columns.get_level_values(1).unique())
                                metric_order_run = [m for m in metric_order_run if m in actual_metrics_in_pivot]


                                yrs_pivot_run_list = sorted([yr_p for yr_p in pivot_final_run.columns.get_level_values(0).unique() if isinstance(yr_p, (int, float, np.number))])

                                if yrs_pivot_run_list and metric_order_run:
                                    final_multi_idx_run = pd.MultiIndex.from_product([yrs_pivot_run_list, metric_order_run], names=['year', 'metric'])
                                    pivot_final_run = pivot_final_run.reindex(columns=final_multi_idx_run).sort_index(axis=1, level=[0, 1])
                                else: pivot_final_run = pd.DataFrame(index=pivot_intermediate_run.index) # Empty if no years or metrics
                                pivot_final_run.index.names = ['Project Name', 'Type']
                            pivot_display_run = pivot_final_run.copy()

                            if not summary_df_run.empty and 'Year' in summary_df_run.columns and yrs_pivot_run_list:
                                total_data_dict_run = {}
                                summary_indexed_run = summary_df_run.set_index('Year')
                                for yr_total_run_loop in yrs_pivot_run_list:
                                    if yr_total_run_loop in summary_indexed_run.index:
                                        s_row_run = summary_indexed_run.loc[yr_total_run_loop]
                                        if 'volume' in metric_order_run: total_data_dict_run[(yr_total_run_loop, 'volume')] = s_row_run.get('Allocated Volume',0)
                                        if 'cost' in metric_order_run: total_data_dict_run[(yr_total_run_loop, 'cost')] = s_row_run.get('Allocated Cost',0.0)
                                        if 'price' in metric_order_run: total_data_dict_run[(yr_total_run_loop, 'price')] = s_row_run.get('Avg. Price',0.0)
                                        if 'margin' in metric_order_run: total_data_dict_run[(yr_total_run_loop, 'margin')] = s_row_run.get('Total Yearly Margin',0.0)
                                
                                if total_data_dict_run :
                                    total_row_df_run = pd.DataFrame(total_data_dict_run, index=pd.MultiIndex.from_tuples([('Total Portfolio', 'All Types')], names=['Project Name', 'Type']))
                                    if isinstance(total_row_df_run.columns, pd.MultiIndex): total_row_df_run.columns.names = ['year', 'metric']
                                    pivot_display_run = pd.concat([pivot_display_run, total_row_df_run]) if not pivot_display_run.empty else total_row_df_run
                            
                            if not pivot_display_run.empty:
                                try:
                                    if isinstance(pivot_display_run.columns, pd.MultiIndex) and len(pivot_display_run.columns.levels) == 2:
                                        pivot_display_run.columns.names = ['year', 'metric']

                                    has_margin_metric = False
                                    if isinstance(pivot_display_run.columns, pd.MultiIndex) and \
                                        'metric' in pivot_display_run.columns.names and \
                                        'margin' in pivot_display_run.columns.get_level_values('metric'):
                                        has_margin_metric = True

                                    if has_margin_metric:
                                        pivot_display_run['Total Margin'] = pivot_display_run.xs('margin', axis=1, level='metric').sum(axis=1)
                                    else:
                                        pivot_display_run['Total Margin'] = 0.0
                                except KeyError as e_xs:
                                    st.warning(f"Detailed Table: Could not calculate 'Total Margin' per project (KeyError: {e_xs}). Setting to 0.")
                                    pivot_display_run['Total Margin'] = 0.0
                                except Exception as e_general_margin_calc:
                                    st.error(f"Detailed Table: Unexpected error calculating 'Total Margin': {e_general_margin_calc}")
                                    pivot_display_run['Total Margin'] = 0.0
                                
                                pivot_display_run = pivot_display_run.fillna(0)
                                formatter_run = {}
                                for col_tuple_fmt_loop in pivot_display_run.columns:
                                    if isinstance(col_tuple_fmt_loop, tuple) and len(col_tuple_fmt_loop) == 2: # MultiIndex columns
                                        metric_fmt_val = col_tuple_fmt_loop[1]
                                        if metric_fmt_val == 'volume': formatter_run[col_tuple_fmt_loop] = '{:,.0f} t'
                                        elif metric_fmt_val == 'cost': formatter_run[col_tuple_fmt_loop] = '€{:,.2f}'
                                        elif metric_fmt_val == 'price': formatter_run[col_tuple_fmt_loop] = lambda x_fmt_val: f'€{x_fmt_val:,.2f}/t' if pd.notna(x_fmt_val) and x_fmt_val != 0 else '-'
                                        elif metric_fmt_val == 'margin': formatter_run[col_tuple_fmt_loop] = '€{:,.2f}'
                                    elif col_tuple_fmt_loop == 'Total Margin': # Single column 'Total Margin'
                                         formatter_run[col_tuple_fmt_loop] = '€{:,.2f}'
                                st.dataframe(pivot_display_run.style.format(formatter_run, na_rep="-"), use_container_width=True)
                            else: st.info("No data for detailed allocation table after processing.")
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
            except Exception as e_gen_run_main: st.error(f"Unexpected error during portfolio calculation or display: {e_gen_run_main}"); st.error(f"Traceback: {traceback.format_exc()}")
        else: st.error("⚠️ Missing critical settings for allocation. Please check sidebar (e.g., planning horizon, project selections, targets).")

# --- Footer ---
st.markdown("---")
try:
    zurich_tz = pytz.timezone('Europe/Zurich')
    st.caption(f"Report generated: {datetime.datetime.now(zurich_tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
except Exception: st.caption(f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Timezone: Server Local)")

