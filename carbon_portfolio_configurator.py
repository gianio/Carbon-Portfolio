import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
        /* --- General Font and Background --- */
        body, .stApp {
            background-color: #F0F2F6; /* Light grey background */
            font-family: 'Calibri', sans-serif; /* Set default font */
        }
        /* Increase base font size for the app */
        body, .stApp, .stApp p, .stApp label, .stApp input, .stApp select, .stApp textarea, .stApp button, .stApp div[data-baseweb="select"] > div {
            font-size: 1.1rem !important; /* Increase font size */
        }

        /* --- Input Field Styling --- */
        .stTextInput input, .stNumberInput input, .stDateInput input, .stTimeInput input, .stSelectbox select, .stMultiselect input[data-baseweb="input"] {
            background-color: #FFFFFF !important; /* White background */
            color: #262730 !important; /* Darker text for readability */
            border: 1px solid #DCDCDC !important; /* Subtle border */
            border-radius: 0.25rem !important; /* Standard radius */
            font-family: 'Calibri', sans-serif !important; /* Ensure font */
            font-size: 1.1rem !important; /* Ensure font size */
        }
        /* Ensure number input label font */
        .stNumberInput label {
             font-family: 'Calibri', sans-serif !important;
             font-size: 1.1rem !important;
        }

        /* --- Slider Styling --- */
        div[data-baseweb="slider"] {
            /* Color of the thumb (circle) */
            & > div:nth-last-child(2) { /* Selector for the thumb */
                 background-color: #A2C244 !important; /* Custom green/yellow */
                 border-color: #A2C244 !important;
            }
            /* Color of the track's filled part */
            & > div:nth-child(2) > div { /* Selector for the filled track */
                background-color: #A2C244 !important;
            }
        }

        /* --- Multiselect Styling --- */
        /* Background of selected items (pills) */
        .stMultiSelect span[data-baseweb="tag"] {
            background-color: #A2C244 !important; /* Custom green/yellow */
            color: #FFFFFF !important; /* White text on the pill */
            font-family: 'Calibri', sans-serif !important; /* Ensure font */
            font-size: 1.0rem !important; /* Slightly smaller font for pills */
            /* Make pills slightly rounder */
            border-radius: 0.5rem !important;
        }
         /* Change border color when multiselect is focused */
        .stMultiSelect [data-baseweb="input"]:focus {
           border: 2px solid #A2C244 !important; /* Thicker border on focus */
           box-shadow: none !important; /* Remove default blue shadow */
        }
        /* Ensure font in dropdown list */
         div[data-baseweb="popover"] ul[role="listbox"] li {
            font-family: 'Calibri', sans-serif !important;
            font-size: 1.1rem !important;
         }


        /* --- Headers and Titles --- */
        h1 { /* Main Title */
            font-weight: bold;
            font-family: 'Calibri', sans-serif !important;
            font-size: 2.5rem !important; /* Larger title */
            color: #31333F;
        }
        h2 { /* Subheaders */
            border-bottom: 2px solid #A2C244; /* Custom green/yellow underline */
            padding-bottom: 8px; /* More space below text */
            margin-top: 30px; /* More space above */
            margin-bottom: 15px; /* More space below */
            font-weight: bold;
            font-family: 'Calibri', sans-serif !important;
            font-size: 1.6rem !important; /* Larger subheaders */
             color: #31333F;
        }
         h3 { /* Sub-subheaders (e.g., Steps) */
            font-weight: bold;
            font-family: 'Calibri', sans-serif !important;
            font-size: 1.3rem !important;
             color: #31333F;
         }

        /* --- General Spacing --- */
        .stApp > header { margin-bottom: 25px; } /* More space below title */
        .stButton>button { margin-top: 15px; } /* Add margin above buttons */

        /* Adjust vertical spacing for number inputs in columns */
        div.stVerticalBlock > div[data-testid="element-container"] > div.stNumberInput {
             margin-bottom: 0px !important; /* Reduce bottom margin */
             padding-bottom: 0px !important;
        }

    </style>
""", unsafe_allow_html=True)
st.title("🌱 Multi-Year Carbon Portfolio Builder")

# --- Data Input ---
df_upload = st.file_uploader("Upload project CSV", type="csv")

# --- Main Logic (Conditional on CSV Upload) ---
if df_upload:
    # --- [PREVIOUS CODE FOR DATA READING AND OVERVIEW - REMAINS THE SAME] ---
    data = pd.read_csv(df_upload)
    required_cols = ['project name', 'project type']
    if not all(col in data.columns for col in required_cols):
        st.error(f"CSV must contain at least the following columns: {', '.join(required_cols)}")
        st.stop()

    st.subheader("Project Overview")
    years = st.number_input("How many years should the portfolio span?", min_value=1, max_value=20, value=6, step=1)
    start_year = 2025
    end_year = start_year + years - 1
    selected_years = list(range(start_year, end_year + 1))

    overview = data.copy()
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in overview.columns]

    if 'priority' not in overview.columns:
        overview['priority'] = 50
    else:
        overview['priority'] = overview['priority'].fillna(50)

    if price_cols:
        for col in price_cols:
            overview[col] = pd.to_numeric(overview[col], errors='coerce')
        overview['avg_price'] = overview[price_cols].mean(axis=1)
        overview_display_cols = ['project name', 'project type']
        if 'description' in overview.columns: overview_display_cols.append('description')
        overview_display_cols.append('avg_price')
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True))
    else:
        overview_display_cols = ['project name', 'project type']
        if 'description' in overview.columns: overview_display_cols.append('description')
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True))
        st.warning(f"No price columns found for years {start_year}-{end_year}. Cannot calculate average price.")
    # --- [END OF PREVIOUS CODE FOR DATA READING AND OVERVIEW] ---


    # --- Portfolio Settings Section ---
    st.subheader("Step 1: Define Portfolio Settings")
    constraint_type = st.radio("Constraint type:", ["Volume Constrained", "Budget Constrained"], key="constraint_type", horizontal=True)
    annual_constraints = {}

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.markdown("**Annual Constraints**") # Keep this title
        # Display yearly inputs
        cols_per_year = st.columns(years if years <= 6 else 6)
        col_idx = 0
        if constraint_type == "Volume Constrained":
            # DELETED markdown text "Enter annual purchase volumes (tonnes):"
            for year in selected_years:
                with cols_per_year[col_idx % len(cols_per_year)]:
                    default_val = 1000
                    annual_constraints[year] = st.number_input(f"{year}", min_value=0, step=100, value=default_val, key=f"vol_{year}", help="Target volume in tonnes for this year")
                col_idx += 1
        else: # Budget Constrained
            # DELETED markdown text "Enter annual budget (€):"
            for year in selected_years:
                 with cols_per_year[col_idx % len(cols_per_year)]:
                    default_val = 10000
                    annual_constraints[year] = st.number_input(f"{year} (€)", min_value=0, step=1000, value=default_val, key=f"bud_{year}", help="Target budget in Euro for this year")
                 col_idx += 1

    with col2:
        st.markdown("**Portfolio Strategy**") # Keep this title
        removal_target = st.slider(f"Target Removal % by {end_year}", 0, 100, 80, key="removal_target") / 100
        transition_speed = st.slider("Transition Speed (1=Slow, 10=Fast)", 1, 10, 5, key="transition_speed")
        removal_preference = st.slider("Removal Preference (1=Natural, 10=Technical)", 1, 10, 5, key="removal_preference", help="Influences allocation order and targets in mixed portfolios")
        # Removed caption, integrated into help text

    # --- [PREVIOUS CODE FOR PROJECT SELECTION - REMAINS THE SAME] ---
    st.subheader("Step 2: Select and Prioritize Projects")
    project_names = sorted(data['project name'].unique().tolist())
    selected_projects = st.multiselect("Select projects to include:", project_names, default=project_names, key="select_proj")
    favorite_projects = st.multiselect("Select favorite projects (+10% priority boost):", selected_projects, key="fav_proj")
    # --- [END OF PREVIOUS CODE FOR PROJECT SELECTION] ---


    if selected_projects:
        selected_df = data[data['project name'].isin(selected_projects)].copy()
        if favorite_projects:
            selected_df['priority'] = selected_df.apply(lambda row: row['priority'] + 10 if row['project name'] in favorite_projects else row['priority'], axis=1)
            selected_df['priority'] = selected_df['priority'].clip(upper=100)

        global_selected_types = selected_df['project type'].unique().tolist()
        global_removal_types = [t for t in global_selected_types if t in ['technical removal', 'natural removal']]
        global_reduction_type = 'reduction' if 'reduction' in global_selected_types else None
        global_has_removals = bool(global_removal_types)
        global_has_reductions = bool(global_reduction_type)

        # --- [PREVIOUS CODE FOR ALLOCATION LOGIC (Year Loop, Phase 1, Phase 2) - REMAINS THE SAME] ---
        # ... (Keep the allocation logic block from the previous response) ...
        portfolio = {year: {} for year in selected_years}
        allocation_warnings = []

        # --- Start Year Loop ---
        for year_idx, year in enumerate(selected_years):
            year_str = str(year)
            volume_col = f"available volume {year_str}"
            price_col = f"price {year_str}"

            if volume_col not in selected_df.columns or price_col not in selected_df.columns:
                allocation_warnings.append(f"Warning for {year}: Missing '{volume_col}' or '{price_col}' column. Skipping.")
                portfolio[year] = {}
                continue

            year_df = selected_df[['project name', 'project type', 'priority', volume_col, price_col]].copy()
            year_df.rename(columns={volume_col: 'volume', price_col: 'price'}, inplace=True)
            year_df['volume'] = pd.to_numeric(year_df['volume'], errors='coerce').fillna(0)
            year_df['price'] = pd.to_numeric(year_df['price'], errors='coerce')
            year_df = year_df[(year_df['volume'] > 0) & pd.notna(year_df['price'])]

            if year_df.empty:
                portfolio[year] = {}
                continue

            annual_limit = annual_constraints.get(year, 0)
            if annual_limit <= 0:
                 portfolio[year] = {}
                 continue

            total_allocated_volume_year = 0.0
            total_allocated_cost_year = 0.0
            allocated_projects_this_year = {}

            year_fraction = (year_idx + 1) / years
            current_removal_target_fraction = removal_target * (year_fraction ** (0.5 + 0.1 * transition_speed))

            current_year_types = year_df['project type'].unique().tolist()
            current_has_removals = any(t in ['technical removal', 'natural removal'] for t in current_year_types)
            current_has_reductions = 'reduction' in current_year_types
            current_only_removals = current_has_removals and not current_has_reductions
            current_only_reductions = not current_has_removals and current_has_reductions

            if current_only_reductions: current_removal_target_fraction = 0.0
            elif current_only_removals: current_removal_target_fraction = 1.0
            elif not current_has_removals and not current_has_reductions:
                 portfolio[year] = {}; continue

            target_removal_alloc = annual_limit * current_removal_target_fraction
            target_reduction_alloc = annual_limit * (1.0 - current_removal_target_fraction)

            year_df.sort_values(by=['priority', 'price'], ascending=[False, True], inplace=True)
            removals_df = year_df[year_df['project type'].isin(global_removal_types)].copy() if current_has_removals else pd.DataFrame()
            reductions_df = year_df[year_df['project type'] == global_reduction_type].copy() if current_has_reductions else pd.DataFrame()

            # --- PHASE 1 ---
            allocated_natural_value = 0.0
            allocated_technical_value = 0.0
            allocated_reduction_value = 0.0

            # 1.a Allocate Removals
            if current_has_removals and not removals_df.empty:
                removal_pref_factor = removal_preference / 10.0
                natural_pref_target = target_removal_alloc * (1.0 - removal_pref_factor)
                technical_pref_target = target_removal_alloc * removal_pref_factor
                removal_order_types = ['natural removal', 'technical removal']
                if removal_preference > 5: removal_order_types.reverse()

                for r_type in removal_order_types:
                    if r_type not in removals_df['project type'].unique(): continue
                    type_df = removals_df[removals_df['project type'] == r_type]
                    for idx, project in type_df.iterrows():
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break
                        project_name = project['project name']
                        available_vol = year_df.loc[idx, 'volume']
                        price = project['price']
                        if available_vol < 1e-6: continue
                        vol_to_allocate = 0.0
                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        cost_of_unit = price if price > 0 else 0
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)

                        if constraint_type == "Volume Constrained":
                            if current_only_removals: vol_to_allocate = min(available_vol, remaining_overall_limit_vol)
                            else:
                                remaining_pref_target_vol = max(0, (natural_pref_target if r_type == 'natural removal' else technical_pref_target) - (allocated_natural_value if r_type == 'natural removal' else allocated_technical_value))
                                vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_pref_target_vol)
                        else: # Budget Constrained
                            if current_only_removals: vol_to_allocate = min(available_vol, affordable_vol_overall)
                            else:
                                remaining_pref_target_cost = max(0, (natural_pref_target if r_type == 'natural removal' else technical_pref_target) - (allocated_natural_value if r_type == 'natural removal' else allocated_technical_value))
                                affordable_vol_pref = remaining_pref_target_cost / (cost_of_unit + 1e-9)
                                vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_pref)

                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate; total_allocated_cost_year += cost
                            if r_type == 'natural removal': allocated_natural_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            else: allocated_technical_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': r_type}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            year_df.loc[idx, 'volume'] -= vol_to_allocate
                    if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                       (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break

            # 1.b Allocate Reductions
            if current_has_reductions and not reductions_df.empty:
                 if not ((constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                         (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)):
                    for idx, project in reductions_df.iterrows():
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break
                        project_name = project['project name']; available_vol = year_df.loc[idx, 'volume']; price = project['price']
                        if available_vol < 1e-6: continue
                        vol_to_allocate = 0.0; cost_of_unit = price if price > 0 else 0
                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)
                        remaining_reduction_target_vol = max(0, target_reduction_alloc - allocated_reduction_value)
                        remaining_reduction_target_cost = max(0, target_reduction_alloc - allocated_reduction_value)
                        affordable_vol_reduction_target = remaining_reduction_target_cost / (cost_of_unit + 1e-9)
                        if constraint_type == "Volume Constrained": vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_reduction_target_vol)
                        else: vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_reduction_target)
                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate; total_allocated_cost_year += cost
                            allocated_reduction_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': global_reduction_type}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            year_df.loc[idx, 'volume'] -= vol_to_allocate

            # --- PHASE 2 ---
            limit_met = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                        (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)
            if not limit_met and annual_limit > 0:
                remaining_projects_df = year_df[year_df['volume'] > 1e-6].sort_values(by=['priority', 'price'], ascending=[False, True])
                if not remaining_projects_df.empty:
                    for idx, project in remaining_projects_df.iterrows():
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break
                        project_name = project['project name']; available_vol = project['volume']; price = project['price']
                        vol_to_allocate = 0.0; cost_of_unit = price if price > 0 else 0
                        if constraint_type == "Volume Constrained":
                            remaining_overall_limit = max(0, annual_limit - total_allocated_volume_year)
                            vol_to_allocate = min(available_vol, remaining_overall_limit)
                        else:
                            remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                            affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9)
                            vol_to_allocate = min(available_vol, affordable_vol_overall)
                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate; total_allocated_cost_year += cost
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate

            # --- Store and Warn ---
            portfolio[year] = allocated_projects_this_year
            final_limit_met = (constraint_type == "Volume Constrained" and abs(total_allocated_volume_year - annual_limit) < 1e-6) or \
                              (constraint_type == "Budget Constrained" and abs(total_allocated_cost_year - annual_limit) < 1e-6) or \
                              (annual_limit <=0)
            if not final_limit_met and annual_limit > 0:
                limit_partially_met = (total_allocated_volume_year > 1e-6) if constraint_type == "Volume Constrained" else (total_allocated_cost_year > 1e-6)
                if not limit_partially_met: allocation_warnings.append(f"Warning {year}: Could not allocate any volume/budget. Target: {annual_limit:.2f}.")
                else:
                     if constraint_type == "Volume Constrained": allocation_warnings.append(f"Warning {year}: Allocated {total_allocated_volume_year:.0f} / {annual_limit:.0f} tonnes (insufficient supply).")
                     else: allocation_warnings.append(f"Warning {year}: Spent €{total_allocated_cost_year:.2f} / €{annual_limit:.2f} budget (insufficient supply).")
        # --- End Year Loop ---

        # --- [PREVIOUS CODE FOR DISPLAYING WARNINGS, PLOTS, SUMMARY, RAW DATA - REMAINS THE SAME, BUT USES UPDATED COLORS/FONTS] ---
        if allocation_warnings:
            st.warning("Allocation Notes & Warnings:")
            for warning in allocation_warnings: st.markdown(f"- {warning}")

        all_types_in_portfolio = set()
        portfolio_data_list = []
        # ... (rest of analysis and plotting code remains the same as previous version) ...
        # --- [ MAKE SURE THE color_map HERE MATCHES THE NEW ONE ] ---
        for year, projects in portfolio.items():
             if projects:
                for name, info in projects.items():
                    all_types_in_portfolio.add(info['type'])
                    portfolio_data_list.append({'year': year, 'project name': name, 'type': info['type'], 'volume': info['volume'], 'price': info['price'], 'cost': info['volume'] * info['price']})

        if not portfolio_data_list:
             st.error("No projects allocated."); st.stop()

        portfolio_df = pd.DataFrame(portfolio_data_list)
        summary_list = []
        for year in selected_years:
            year_data = portfolio_df[portfolio_df['year'] == year]
            total_volume = year_data['volume'].sum(); total_cost = year_data['cost'].sum()
            avg_price = total_cost / (total_volume + 1e-9)
            volume_by_type = year_data.groupby('type')['volume'].sum()
            summary_entry = {'Year': year, 'Total Volume (tonnes)': total_volume, 'Total Cost (€)': total_cost, 'Average Price (€/tonne)': avg_price, 'Target Constraint': annual_constraints.get(year, 0)}
            for proj_type in (global_removal_types + ([global_reduction_type] if global_reduction_type else [])):
                  summary_entry[f'Volume {proj_type.capitalize()}'] = volume_by_type.get(proj_type, 0)
            summary_list.append(summary_entry)
        summary_df = pd.DataFrame(summary_list)

        st.subheader("Portfolio Composition & Price Over Time")
        fig = make_subplots(specs=[[[{"secondary_y": True}]])

        # --- >>> USE UPDATED COLOR MAP HERE <<< ---
        color_map = {
            'technical removal': '#64B5F6', # Blueish
            'natural removal': '#81C784', # Greenish
            'reduction': '#B0BEC5', # Greyish
        }
        default_color = '#D3D3D3'
        # --- >>> END OF UPDATED COLOR MAP <<< ---

        plot_types = sorted([t for t in all_types_in_portfolio if t in color_map])
        for type_name in plot_types:
            type_volume_col = f'Volume {type_name.capitalize()}'
            if type_volume_col in summary_df.columns:
                type_volume = summary_df[type_volume_col]
                fig.add_trace(go.Bar(x=summary_df['Year'], y=type_volume, name=type_name.capitalize(), marker_color=color_map.get(type_name, default_color)), secondary_y=False)
        fig.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Average Price (€/tonne)'], name='Avg Price (€/tonne)', marker=dict(symbol='circle'), line=dict(color='#546E7A')), secondary_y=True)
        fig.update_layout(xaxis_title='Year', yaxis_title='Volume (tonnes)', yaxis2_title='Average Price (€/tonne)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero'))
        st.plotly_chart(fig, use_container_width=True)

        if end_year in summary_df['Year'].values:
             final_year_summary = summary_df[summary_df['Year'] == end_year].iloc[0]
             final_tech = final_year_summary.get(f'Volume Technical removal', 0) if 'Volume Technical removal' in final_year_summary else 0
             final_nat = final_year_summary.get(f'Volume Natural removal', 0) if 'Volume Natural removal' in final_year_summary else 0
             final_total_removal = final_tech + final_nat
             final_total = final_year_summary['Total Volume (tonnes)']
             achieved_removal_perc = (final_total_removal / (final_total + 1e-9)) * 100
             if global_has_removals: st.metric(label=f"Achieved Removal % in {end_year}", value=f"{achieved_removal_perc:.2f}%")
             elif not selected_projects: st.info("No projects selected.")
             else: st.info("No removal projects selected.")
        else: st.info(f"No data for final year ({end_year}).")

        st.subheader("Yearly Summary")
        summary_display_df = summary_df.copy()
        summary_display_df['Target'] = summary_display_df['Target Constraint']
        summary_display_df['Achieved Volume'] = summary_display_df['Total Volume (tonnes)']
        summary_display_df['Achieved Cost (€)'] = summary_display_df['Total Cost (€)']
        summary_display_df['Avg Price (€/tonne)'] = summary_display_df['Average Price (€/tonne)']
        if constraint_type == "Volume Constrained":
            summary_display_df.rename(columns={'Target': 'Target Volume'}, inplace=True)
            display_cols = ['Year', 'Target Volume', 'Achieved Volume', 'Achieved Cost (€)', 'Avg Price (€/tonne)']
        else:
             summary_display_df.rename(columns={'Target': 'Target Budget (€)'}, inplace=True)
             display_cols = ['Year', 'Target Budget (€)', 'Achieved Cost (€)', 'Achieved Volume', 'Avg Price (€/tonne)']
        for proj_type in plot_types:
             col_name = f'Volume {proj_type.capitalize()}'
             if col_name in summary_display_df.columns:
                  display_cols.append(col_name)
                  summary_display_df[col_name] = summary_display_df[col_name].map('{:,.0f}'.format)
        for col in ['Target Budget (€)', 'Achieved Cost (€)', 'Avg Price (€/tonne)']:
             if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('{:,.2f}'.format)
        for col in ['Target Volume', 'Achieved Volume']:
              if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('{:,.0f}'.format)
        st.dataframe(summary_display_df[display_cols].set_index('Year'))

        st.subheader("Detailed Allocation Data")
        if st.checkbox("Show raw project allocations by year", key="show_raw"):
             display_portfolio_df = portfolio_df.copy()
             display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.2f}'.format)
             display_portfolio_df['price'] = display_portfolio_df['price'].map('{:,.2f}'.format)
             display_portfolio_df['cost'] = display_portfolio_df['cost'].map('{:,.2f}'.format)
             st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']))
        # --- [ END OF DISPLAY SECTION ] ---

    elif df_upload:
         st.warning("Please select at least one project in Step 2 to build the portfolio.")
