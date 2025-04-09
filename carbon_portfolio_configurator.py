import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # Added for potential NaN checks if needed

# --- Define a new color palette ---
primary_color = "#2E7D32"  # Dark Green
secondary_color = "#4CAF50" # Medium Green
accent_color = "#8BC34A"   # Light Green
background_color = "#F0F8F0" # Very Light Green
text_color = "#212121"     # Dark Grey

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide") # Optional: Use wider layout
st.markdown(f"""
    <style>
    body {{
        background-color: {background_color};
        font-family: sans-serif; /* Default to system sans-serif, Calibri will be applied more specifically */
        color: {text_color};
        line-height: 1.4;
    }}
    .stApp {{
        background-color: {background_color};
    }}
    .stApp > header {{
        margin-bottom: 1.5rem;
        padding: 1rem 0;
        background-color: rgba({primary_color_rgb(primary_color)}, 0.05); /* Subtle header background */
        border-bottom: 1px solid {accent_color};
    }}
    h1 {{
        color: {primary_color};
        font-family: 'Calibri', sans-serif;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }}
    h2 {{
        color: {primary_color};
        font-family: 'Calibri', sans-serif;
        font-size: 1.8rem;
        border-bottom: 2px solid {secondary_color};
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }}
    h3 {{
        color: {primary_color};
        font-family: 'Calibri', sans-serif;
        font-size: 1.4rem;
        margin-top: 1.5rem;
    }}
    p, div, stText, stMarkdown, stCaption, stNumberInput label, stSlider label, stFileUploader label, stMultiSelect label, stRadio label, stCheckbox label, .streamlit-expander {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1.1rem !important;
        color: {text_color} !important;
    }}
    .streamlit-expander-content {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1.1rem !important;
        color: {text_color} !important;
        border-left: 0.2rem solid {accent_color};
        padding-left: 1rem;
        margin-top: 0.5rem;
    }}
    .stButton > button {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1.1rem !important;
        color: {text_color} !important;
        border: 1px solid {secondary_color} !important;
        background-color: {background_color} !important;
        border-radius: 0.3rem;
        padding: 0.6rem 1rem;
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    .stButton > button:hover {{
        background-color: {secondary_color} !important;
        color: white !important;
    }}
    .stSlider > div[data-baseweb="slider"] > div[role="slider"]::before {{
        background-color: {secondary_color} !important;
    }}
    .stSlider > div[data-baseweb="slider"] > div[role="slider"] > span {{
        background-color: {primary_color} !important;
        border-color: {primary_color} !important;
    }}
    .stNumberInput > div > div > input, .stSelectbox > div > div > div > button, .stMultiSelect > div > div > div > button, .stFileUploader > div > div:first-child > div:first-child > label {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1.1rem !important;
        color: {text_color} !important;
        border: 1px solid {accent_color} !important;
        border-radius: 0.3rem;
        padding: 0.5rem;
        background-color: {background_color};
    }}
    .stRadio > label, .stCheckbox > label {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1.1rem !important;
        color: {text_color} !important;
    }}
    .stDataFrame {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1rem !important;
        color: {text_color} !important;
        border: 1px solid {accent_color};
        border-radius: 0.3rem;
    }}
    .stDataFrame tr th {{
        background-color: {accent_color} !important;
        color: {text_color} !important;
        padding: 0.6rem;
        font-weight: bold;
    }}
    .stDataFrame tr td {{
        padding: 0.5rem;
    }}
    .stMetric {{
        background-color: white !important;
        border: 1px solid {accent_color} !important;
        padding: 1rem !important;
        border-radius: 0.3rem !important;
        box-shadow: 0 0.1rem 0.3rem rgba(0, 0, 0, 0.05);
    }}
    .stMetricLabel {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1rem !important;
        color: {secondary_color} !important;
    }}
    .stMetricValue {{
        font-family: 'Calibri', sans-serif !important;
        font-size: 1.4rem !important;
        color: {primary_color} !important;
    }}
    .stWarning {{
        color: #FFC107;
        background-color: #FFFDE7;
        border-left: 0.3rem solid #FFEB3B;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0.3rem;
    }}
    .stError {{
        color: #D32F2F;
        background-color: #FFEBEE;
        border-left: 0.3rem solid #F44336;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0.3rem;
    }}
    .stInfo {{
        color: #1976D2;
        background-color: #E3F2FD;
        border-left: 0.3rem solid #2196F3;
        padding: 0.8rem;
        margin-bottom: 1rem;
        border-radius: 0.3rem;
    }}
    </style>
""", unsafe_allow_html=True)

def primary_color_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

st.title("🌱 Multi-Year Carbon Portfolio Builder")

# --- Data Input ---
df_upload = st.file_uploader("Upload project CSV", type="csv")

# --- Main Logic (Conditional on CSV Upload) ---
if df_upload:
    data = pd.read_csv(df_upload)
    # Basic check for essential columns (can be expanded)
    required_cols = ['project name', 'project type']
    if not all(col in data.columns for col in required_cols):
        st.error(f"CSV must contain at least the following columns: {', '.join(required_cols)}")
        st.stop() # Stop execution if essential columns are missing

    # --- Project Overview Section ---
    st.subheader("Project Overview")
    years = st.number_input("How many years should the portfolio span?", min_value=1, max_value=20, value=6, step=1)
    start_year = 2025 # Define start year explicitly (Adjust if needed)
    end_year = start_year + years - 1
    selected_years = list(range(start_year, end_year + 1))

    overview = data.copy()
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in overview.columns]

    # Ensure 'priority' column exists, fill with a default if not (e.g., 50 or median)
    if 'priority' not in overview.columns:
        overview['priority'] = 50 # Assign a neutral default priority
        # st.info("No 'priority' column found in CSV. Assigning a default priority of 50 to all projects.") # Less verbose
    else:
        # Fill NaN priorities with default
        overview['priority'] = overview['priority'].fillna(50)

    # Calculate average price only if price columns exist
    if price_cols:
        # Ensure price columns are numeric, coerce errors to NaN
        for col in price_cols:
            overview[col] = pd.to_numeric(overview[col], errors='coerce')
        overview['avg_price'] = overview[price_cols].mean(axis=1)
        overview_display_cols = ['project name', 'project type']
        if 'description' in overview.columns:
            overview_display_cols.append('description')
        overview_display_cols.append('avg_price')
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True))
    else:
        overview_display_cols = ['project name', 'project type']
        if 'description' in overview.columns:
            overview_display_cols.append('description')
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']).reset_index(drop=True))
        st.warning(f"No price columns found for years {start_year}-{end_year} (e.g., 'price {start_year}'). Cannot calculate average price.")


    # --- Portfolio Settings Section ---
    st.subheader("Step 1: Define Portfolio Settings")
    constraint_type = st.radio("Select constraint type:", ["Volume Constrained", "Budget Constrained"], key="constraint_type", horizontal=True)
    annual_constraints = {}

    # Use columns for layout
    col1, col2 = st.columns([0.6, 0.4]) # Adjust column width ratio if needed

    with col1:
        st.markdown("**Annual Constraints:**")
        # Display yearly inputs in a more compact way if many years
        cols_per_year = st.columns(years if years <= 6 else 6) # Max 6 columns for inputs
        col_idx = 0
        if constraint_type == "Volume Constrained":
            st.markdown("Enter annual purchase volumes (tonnes):")
            # REMOVED global_volume input
            for year in selected_years:
                with cols_per_year[col_idx % len(cols_per_year)]:
                    # Use year as default value for easier debugging if needed, otherwise 1000
                    default_val = 1000
                    annual_constraints[year] = st.number_input(f"{year}", min_value=0, step=100, value=default_val, key=f"vol_{year}", label_visibility="visible") # Show year label
                col_idx += 1
        else:
            st.markdown("Enter annual budget (€):")
            # REMOVED global_budget input
            for year in selected_years:
                with cols_per_year[col_idx % len(cols_per_year)]:
                    # Use year*10 as default value for easier debugging if needed, otherwise 10000
                    default_val = 10000
                    annual_constraints[year] = st.number_input(f"{year} (€)", min_value=0, step=1000, value=default_val, key=f"bud_{year}", label_visibility="visible") # Show year label
                col_idx += 1

    with col2:
        st.markdown("**Portfolio Strategy:**")
        removal_target = st.slider(f"Target Removal % by {end_year}", 0, 100, 80, key="removal_target") / 100
        transition_speed = st.slider("Transition Speed (1=Slow, 10=Fast)", 1, 10, 5, key="transition_speed")
        # Make label clearer about slider direction
        removal_preference = st.slider("Removal Preference (1=Natural Favored, 10=Technical Favored)", 1, 10, 5, key="removal_preference")
        st.caption("Preference influences allocation order and targets in mixed portfolios.")


    # --- Project Selection and Prioritization Section ---
    st.subheader("Step 2: Select and Prioritize Projects")
    project_names = sorted(data['project name'].unique().tolist()) # Sort alphabetically
    selected_projects = st.multiselect("Select projects to include:", project_names, default=project_names, key="select_proj")
    favorite_projects = st.multiselect("Select favorite projects (+10% priority boost):", selected_projects, key="fav_proj")

    if selected_projects:
        selected_df = data[data['project name'].isin(selected_projects)].copy()

        # Apply favorite boost (ensure priority column exists from earlier check)
        if favorite_projects:
            selected_df['priority'] = selected_df.apply(
                lambda row: row['priority'] + 10 if row['project name'] in favorite_projects else row['priority'],
                axis=1
            )
            selected_df['priority'] = selected_df['priority'].clip(upper=100) # Cap priority at 100

        # Identify available project types globally selected
        global_selected_types = selected_df['project type'].unique().tolist()
        global_removal_types = [t for t in global_selected_types if t in ['technical removal', 'natural removal']]
        global_reduction_type = 'reduction' if 'reduction' in global_selected_types else None
        global_has_removals = bool(global_removal_types)
        global_has_reductions = bool(global_reduction_type)


        # --- Allocation Logic ---
        portfolio = {year: {} for year in selected_years} # Stores { project_name: { volume: v, price: p, type: t } }
        allocation_warnings = []

        # --- Start Year Loop ---
        for year_idx, year in enumerate(selected_years):
            year_str = str(year)
            volume_col = f"available volume {year_str}"
            price_col = f"price {year_str}"

            # Check if essential columns exist for the current year
            if volume_col not in selected_df.columns or price_col not in selected_df.columns:
                allocation_warnings.append(f"Warning for {year}: Missing '{volume_col}' or '{price_col}' column. Skipping allocation for this year.")
                portfolio[year] = {}
                continue

            # Prepare DataFrame for this year's allocation
            year_df = selected_df[['project name', 'project type', 'priority', volume_col, price_col]].copy()
            year_df.rename(columns={volume_col: 'volume', price_col: 'price'}, inplace=True)

            # Convert volume and price to numeric, handle errors/NaNs
            year_df['volume'] = pd.to_numeric(year_df['volume'], errors='coerce').fillna(0)
            year_df['price'] = pd.to_numeric(year_df['price'], errors='coerce')

            # Filter out projects with zero volume or invalid price for the year
            year_df = year_df[(year_df['volume'] > 0) & pd.notna(year_df['price'])]

            if year_df.empty:
                # allocation_warnings.append(f"Note for {year}: No projects with available volume and valid price found.") # Less alarming
                portfolio[year] = {}
                continue

            # --- Year Specific Calculations ---
            annual_limit = annual_constraints.get(year, 0)
            if annual_limit <= 0:
                portfolio[year] = {}
                continue # Skip year if constraint is zero

            total_allocated_volume_year = 0.0
            total_allocated_cost_year = 0.0
            allocated_projects_this_year = {}

            # Calculate target split for removals/reductions based on transition
            year_fraction = (year_idx + 1) / years
            current_removal_target_fraction = removal_target * (year_fraction ** (0.5 + 0.1 * transition_speed))

            # Determine available types FOR THIS YEAR and portfolio type
            current_year_types = year_df['project type'].unique().tolist()
            current_has_removals = any(t in ['technical removal', 'natural removal'] for t in current_year_types)
            current_has_reductions = 'reduction' in current_year_types
            current_only_removals = current_has_removals and not current_has_reductions
            current_only_reductions = not current_has_removals and current_has_reductions

            if current_only_reductions:
                current_removal_target_fraction = 0.0
            elif current_only_removals:
                current_removal_target_fraction = 1.0
            elif not current_has_removals and not current_has_reductions:
                # Should not happen if year_df is not empty, but safety check
                portfolio[year] = {}
                continue

            target_removal_alloc = annual_limit * current_removal_target_fraction
            target_reduction_alloc = annual_limit * (1.0 - current_removal_target_fraction)

            # Sort projects: High priority first, then low price
            year_df.sort_values(by=['priority', 'price'], ascending=[False, True], inplace=True)

            # Get DFs for removals and reductions for this year
            removals_df = year_df[year_df['project type'].isin(global_removal_types)].copy() if current_has_removals else pd.DataFrame()
            reductions_df = year_df[year_df['project type'] == global_reduction_type].copy() if current_has_reductions else pd.DataFrame()

            # --- PHASE 1: Targeted Allocation ---
            allocated_natural_value = 0.0 # Track allocated volume/cost for each removal type in Phase 1
            allocated_technical_value = 0.0
            allocated_reduction_value = 0.0

            # 1.a Allocate Removals (respecting preference)
            if current_has_removals and not removals_df.empty:
                removal_pref_factor = removal_preference / 10.0
                # Calculate the *ideal* target split based on preference (volume or cost depends on constraint)
                # This target guides Phase 1 allocation in mixed portfolios
                natural_pref_target = target_removal_alloc * (1.0 - removal_pref_factor)
                technical_pref_target = target_removal_alloc * removal_pref_factor

                removal_order_types = ['natural removal', 'technical removal']
                if removal_preference > 5:
                    removal_order_types.reverse()

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

                        # Define remaining limits
                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        cost_of_unit = price if price > 0 else 0
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)

                        if constraint_type == "Volume Constrained":
                            if current_only_removals:
                                vol_to_allocate = min(available_vol, remaining_overall_limit_vol)
                            else: # Mixed portfolio - Limit by overall AND specific preference target for this type
                                if r_type == 'natural removal':
                                    remaining_pref_target_vol = max(0, natural_pref_target - allocated_natural_value)
                                else: # technical removal
                                    remaining_pref_target_vol = max(0, technical_pref_target - allocated_technical_value)
                                vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_pref_target_vol)
                        else: # Budget Constrained
                            if current_only_removals:
                                vol_to_allocate = min(available_vol, affordable_vol_overall)
                            else: # Mixed portfolio - Limit by overall AND specific preference target for this type
                                if r_type == 'natural removal':
                                    remaining_pref_target_cost = max(0, natural_pref_target - allocated_natural_value)
                                    affordable_vol_pref = remaining_pref_target_cost / (cost_of_unit + 1e-9)
                                else: # technical removal
                                    remaining_pref_target_cost = max(0, technical_pref_target - allocated_technical_value)
                                    affordable_vol_pref = remaining_pref_target_cost / (cost_of_unit + 1e-9)
                                vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_pref)

                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            # Update specific removal type tracking value
                            if r_type == 'natural removal': allocated_natural_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            else: allocated_technical_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            # Store allocation
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': r_type}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            year_df.loc[idx, 'volume'] -= vol_to_allocate # Decrement remaining volume
                    # Check limit after each type
                    if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                       (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break

            # 1.b Allocate Reductions (Only if reductions exist and limit not met)
            if current_has_reductions and not reductions_df.empty:
                if not ((constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                        (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)):
                    for idx, project in reductions_df.iterrows():
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break
                        project_name = project['project name']
                        available_vol = year_df.loc[idx, 'volume']
                        price = project['price']
                        if available_vol < 1e-6: continue
                        vol_to_allocate = 0.0
                        cost_of_unit = price if price > 0 else 0

                        # Define remaining limits
                        remaining_overall_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        remaining_overall_limit_cost = max(0, annual_limit - total_allocated_cost_year)
                        affordable_vol_overall = remaining_overall_limit_cost / (cost_of_unit + 1e-9)
                        # Define remaining target for reductions in Phase 1
                        remaining_reduction_target_vol = max(0, target_reduction_alloc - allocated_reduction_value)
                        remaining_reduction_target_cost = max(0, target_reduction_alloc - allocated_reduction_value) # Assuming target is cost if budget constrained
                        affordable_vol_reduction_target = remaining_reduction_target_cost / (cost_of_unit + 1e-9)


                        if constraint_type == "Volume Constrained":
                            vol_to_allocate = min(available_vol, remaining_overall_limit_vol, remaining_reduction_target_vol)
                        else: # Budget Constrained
                            vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_reduction_target)

                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            allocated_reduction_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': global_reduction_type}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            year_df.loc[idx, 'volume'] -= vol_to_allocate

            # --- PHASE 2: Gap Filling ---
            limit_met = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                        (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)
            if not limit_met and annual_limit > 0:
                remaining_projects_df = year_df[year_df['volume'] > 1e-6].sort_values(by=['priority', 'price'], ascending=[False, True])
                if not remaining_projects_df.empty:
                    for idx, project in remaining_projects_df.iterrows():
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6): break
                        project_name = project['project name']
                        available_vol = project['volume']
                        price = project['price']
                        vol_to_allocate = 0.0
                        cost_of_unit = price if price > 0 else 0

                        if constraint_type == "Volume Constrained":
                            remaining_overall_limit = max(0, annual_limit - total_allocated_volume_year)
                            vol_to_allocate = min(available_vol, remaining_overall_limit)
                        else: # Budget Constrained
                            remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                            affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9)
                            vol_to_allocate = min(available_vol, affordable_vol_overall)

                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            if project_name not in allocated_projects_this_year: allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            # Optional: Decrement year_df volume here too if needed elsewhere, but not strictly necessary for Phase 2 logic itself.
                            # year_df.loc[idx, 'volume'] -= vol_to_allocate

            # --- Store and Warn ---
            portfolio[year] = allocated_projects_this_year
            final_limit_met = (constraint_type == "Volume Constrained" and abs(total_allocated_volume_year - annual_limit) < 1e-6) or \
                              (constraint_type == "Budget Constrained" and abs(total_allocated_cost_year - annual_limit) < 1e-6) or \
                              (annual_limit <=0) # Consider limit met if target was 0

            if not final_limit_met and annual_limit > 0:
                limit_partially_met = (total_allocated_volume_year > 1e-6) if constraint_type == "Volume Constrained" else (total_allocated_cost_year > 1e-6)
                if not limit_partially_met:
                    allocation_warnings.append(f"Warning for {year}: Could not allocate *any* volume/budget towards the target of {annual_limit:.2f}. Check project availability and prices.")
                else:
                    if constraint_type == "Volume Constrained": allocation_warnings.append(f"Warning for {year}: Could only allocate {total_allocated_volume_year:.2f} tonnes out of target {annual_limit:.0f} due to insufficient project availability.")
                    else: allocation_warnings.append(f"Warning for {year}: Could only spend €{total_allocated_cost_year:.2f} out of target budget €{annual_limit:.2f} due to insufficient affordable project volume.")
        # --- End Year Loop ---

        # --- Display Warnings ---
        if allocation_warnings:
            st.warning("Allocation Notes & Warnings:")
            for warning in allocation_warnings:
                st.markdown(f"- {warning}")

        # --- Portfolio Analysis and Visualization ---
        all_types_in_portfolio = set()
        portfolio_data_list = []
        for year, projects in portfolio.items():
            if projects: # Only process years where allocation happened
                for name, info in projects.items():
                    all_types_in_portfolio.add(info['type'])
                    portfolio_data_list.append({
                        'year': year,
                        'project name': name,
                        'type': info['type'],
                        'volume': info['volume'],
                        'price': info['price'],
                        'cost': info['volume'] * info['price']
                    })

        if not portfolio_data_list:
            st.error("No projects could be allocated based on the selected criteria and available data.")
            st.stop()

        portfolio_df = pd.DataFrame(portfolio_data_list)

        # Aggregations for plotting and summary
        summary_dict = {}
        for year in selected_years:
            year_data = portfolio_df[portfolio_df['year'] == year]
            total_volume = year_data['volume'].sum()
            total_cost = year_data['cost'].sum()
            natural_removal = year_data[year_data['type'] == 'natural removal']['volume'].sum()
            technical_removal = year_data[year_data['type'] == 'technical removal']['volume'].sum()
            reduction = year_data[year_data['type'] == 'reduction']['volume'].sum()

            summary_dict[year] = {
                'Total Volume': total_volume,
                'Natural Removal': natural_removal,
                'Technical Removal': technical_removal,
                'Reductions': reduction,
                'Total Cost': total_cost
            }

        summary_df_new = pd.DataFrame.from_dict(summary_dict, orient='index')
        summary_df_new.index.name = 'Year'
        summary_df_transposed = summary_df_new.T.reset_index().rename(columns={'index': 'Metric'})

        st.subheader("Portfolio Summary by Year")
        st.dataframe(summary_df_transposed.style.format({'Total Volume': '{:,.0f}',
                                                        'Natural Removal': '{:,.0f}',
                                                        'Technical Removal': '{:,.0f}',
                                                        'Reductions': '{:,.0f}',
                                                        'Total Cost': '{:,.2f}'}))


        st.subheader("Portfolio Composition & Price Over Time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Define UPDATED consistent colors
        color_map = {
            'technical removal': '#64B5F6', # Blueish
            'natural removal': '#81C784', # Greenish
            'reduction': '#B0BEC5', # Greyish
                # Add more types and colors if needed
        }
        default_color = '#D3D3D3' # Light Grey for unknown types

        # Plot types present in the actual allocation
        plot_types = sorted([t for t in all_types_in_portfolio if t in color_map])

        for type_name in plot_types:
            type_volume_col = f'Volume {type_name.capitalize()}'
            if type_volume_col in summary_df_new.columns:
                type_volume = summary_df_new[type_volume_col]
                fig.add_trace(go.Bar(x=summary_df_new.index, y=type_volume, name=type_name.capitalize(), marker_color=color_map.get(type_name, default_color)), secondary_y=False)

        if 'Total Cost' in summary_df_new.columns and 'Total Volume' in summary_df_new.columns:
            avg_price = summary_df_new['Total Cost'] / (summary_df_new['Total Volume'] + 1e-9)
            fig.add_trace(go.Scatter(x=summary_df_new.index, y=avg_price, name='Avg Price (€/tonne)', marker=dict(symbol='circle'), line=dict(color='#546E7A')), secondary_y=True) # Darker Blue Grey for price line

        fig.update_layout(xaxis_title='Year', yaxis_title='Volume (tonnes)', yaxis2_title='Average Price (€/tonne)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero'))
        st.plotly_chart(fig, use_container_width=True)

        # Final Year Removal %
        if end_year in summary_df_new.index:
            final_year_summary = summary_df_new.loc[end_year]
            final_tech = final_year_summary.get('Technical Removal', 0)
            final_nat = final_year_summary.get('Natural Removal', 0)
            final_total_removal = final_tech + final_nat
            final_total = final_year_summary.get('Total Volume', 0)
            achieved_removal_perc = (final_total_removal / (final_total + 1e-9)) * 100
            if global_has_removals: st.metric(label=f"Achieved Removal % in {end_year}", value=f"{achieved_removal_perc:.2f}%")
            elif not selected_projects: st.info("No projects selected.")
            else: st.info("No removal projects selected.")
        else:
            st.info(f"No data allocated for the final year ({end_year}).")


        st.subheader("Detailed Allocation Data")
        if st.checkbox("Show raw project allocations by year", key="show_raw"):
            display_portfolio_df = portfolio_df.copy()
            display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.2f}'.format)
            display_portfolio_df['price'] = display_portfolio_df['price'].map('{:,.2f}'.format)
            display_portfolio_df['cost'] = display_portfolio_df['cost'].map('{:,.2f}'.format)
            st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']))

    elif df_upload:
        st.warning("Please select at least one project in Step 2")
