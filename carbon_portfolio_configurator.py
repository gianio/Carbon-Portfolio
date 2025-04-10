import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np # Added for potential NaN checks if needed

# --- Define a nice green color palette ---
primary_green = "#D7F0C8" # Light Green
secondary_green = "#D7F0C8" # Medium Green
background_green = "#D7F0C8" # Very Light Green
text_green = "#33691E" # Dark Green
accent_green = "#D7F0C8" # Lime Green

# --- Streamlit App Configuration ---
st.set_page_config(layout="centered") # Optional: Use wider layout
st.markdown(f"""
    <style>
    body {{
        background-color: {background_green};
        font-family: Calibri, sans-serif;
        font-size: 20.8px; /* 16px * 1.3 */
        color: {text_green};
    }}
    .stApp {{
        background-color: {background_green};
    }}
    .stApp > header {{
        margin-bottom: 13px; /* 10px * 1.3 */
    }}
    h1 {{
        color: {secondary_green};
        font-family: Calibri, sans-serif;
        font-size: 39px; /* 30px * 1.3 */
    }}
    h2 {{
        border-bottom: 2.6px solid {accent_green}; /* 2px * 1.3 */
        padding-bottom: 6.5px; /* 5px * 1.3 */
        margin-top: 26px; /* 20px * 1.3 */
        font-family: Calibri, sans-serif;
        font-size: 31.2px; /* 24px * 1.3 */
        color: {secondary_green};
    }}
    h3 {{
        font-family: Calibri, sans-serif;
        font-size: 26px; /* 20px * 1.3 */
        color: {secondary_green};
    }}
    p, div, stText, stMarkdown, stCaption, stNumberInput label, stSlider label, stFileUploader label, stMultiSelect label, stRadio label, stCheckbox label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important; /* 16px * 1.3 */
        color: {text_green} !important;
    }}
    .streamlit-expander {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
    }}
    .streamlit-expander-content {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
    }}
    .stButton > button {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {secondary_green} !important;
        background-color: {background_green} !important;
        &:hover {{
            background-color: {primary_green} !important;
            color: white !important;
        }}
    }}
    .stSlider > div[data-baseweb="slider"] > div[role="slider"]::before {{
        background-color: {primary_green} !important;
    }}
    .stSlider > div[data-baseweb="slider"] > div[role="slider"] > span {{
        background-color: {secondary_green} !important;
        border-color: {secondary_green} !important;
    }}
    .stNumberInput > div > div > input {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
    }}
    .stSelectbox > div > div > div > button {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
        background-color: {background_green} !important;
    }}
    .stMultiSelect > div > div > div > button {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
        background-color: {background_green} !important;
    }}
    .stRadio > label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
    }}
    .stCheckbox > label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
    }}
    .stFileUploader > div > div:first-child > div:first-child > label {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
        background-color: {background_green} !important;
    }}
    .stDataFrame {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {text_green} !important;
        border-color: {accent_green} !important;
    }}
    .stDataFrame tr th {{
        background-color: {accent_green} !important;
        color: {text_green} !important;
    }}
    .stMetric {{
        background-color: {background_green} !important;
        border: 1px solid {accent_green} !important;
        padding: 15px !important;
        border-radius: 5px !important;
    }}
    .stMetricLabel {{
        font-family: Calibri, sans-serif !important;
        font-size: 20.8px !important;
        color: {secondary_green} !important;
    }}
    .stMetricValue {{
        font-family: Calibri, sans-serif !important;
        font-size: 26px !important;
        color: {text_green} !important;
    }}
    </style>
""", unsafe_allow_html=True)

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

    # --- Data Preparation for the Year ---
    print(f"\n--- Allocating for Year: {year} ---")
    if volume_col not in selected_df.columns or price_col not in selected_df.columns:
        allocation_warnings.append(f"Warning for {year}: Missing '{volume_col}' or '{price_col}' column. Skipping.")
        portfolio[year] = {}
        continue

    year_df = selected_df[['project name', 'project type', 'priority', volume_col, price_col]].copy()
    year_df.rename(columns={volume_col: 'available_volume', price_col: 'price'}, inplace=True) # Use 'available_volume'

    # Convert types, handle errors, fill NaNs
    year_df['available_volume'] = pd.to_numeric(year_df['available_volume'], errors='coerce').fillna(0)
    year_df['price'] = pd.to_numeric(year_df['price'], errors='coerce') # Keep NaN for invalid prices
    # Treat NaN priority as non-prioritized (e.g., 0)
    year_df['priority'] = pd.to_numeric(year_df['priority'], errors='coerce').fillna(0) 

    # Filter out projects unusable for the year
    year_df = year_df[(year_df['available_volume'] > 1e-6) & (pd.notna(year_df['price'])) & (year_df['price'] >= 0)] # Ensure positive price too? Added >=0

    if year_df.empty:
        print(f"  No valid projects found for {year}. Skipping.")
        portfolio[year] = {}
        continue

    # --- Get Annual Limit ---
    annual_limit = annual_constraints.get(year, 0)
    if annual_limit <= 0:
        print(f"  Annual limit for {year} is {annual_limit}. Skipping.")
        portfolio[year] = {}
        continue
    print(f"  Annual Limit ({constraint_type}): {annual_limit}")

    # --- Initialize Allocation Tracking ---
    total_allocated_volume_year = 0.0
    total_allocated_cost_year = 0.0
    # Stores the actual allocated volume per project for the year
    allocated_projects_this_year = {} # { project_name: {'volume': v, 'price': p, 'type': t} }
    # Keep track of remaining volume per project during allocation
    volume_tracker = year_df.set_index('project name')['available_volume'].to_dict()


    # --- Split into Prioritized and Non-Prioritized Groups ---
    prioritized_df = year_df[year_df['priority'] > 0].copy()
    non_prio_df = year_df[year_df['priority'] <= 0].copy() # Includes 0 and previously NaN

    # --- Sort the Groups ---
    prioritized_df.sort_values(by=['priority', 'price'], ascending=[False, True], inplace=True)
    non_prio_df.sort_values(by=['price'], ascending=[True], inplace=True) # Sort by price for budget mode & consistency

    print(f"  Prioritized projects count: {len(prioritized_df)}")
    print(f"  Non-prioritized projects count: {len(non_prio_df)}")

    # --- Stage 1: Allocate Prioritized Projects Sequentially ---
    print("  Stage 1: Processing prioritized projects...")
    if not prioritized_df.empty:
        for idx, project in prioritized_df.iterrows():
            # Check if limit is already met
            limit_met = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                        (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)
            if limit_met:
                print("    Limit reached during prioritized allocation.")
                break

            project_name = project['project name']
            available_vol = volume_tracker.get(project_name, 0) # Get current available volume
            price = project['price']
            cost_of_unit = price if price > 0 else 0 # Avoid negative cost influence

            if available_vol < 1e-6: continue # Should not happen if filtered initially, but safe check

            vol_to_allocate = 0.0
            # Calculate how much can be allocated based on constraint
            if constraint_type == "Volume Constrained":
                remaining_limit = max(0, annual_limit - total_allocated_volume_year)
                vol_to_allocate = min(available_vol, remaining_limit)
            else: # Budget Constrained
                remaining_limit = max(0, annual_limit - total_allocated_cost_year)
                affordable_vol = remaining_limit / (cost_of_unit + 1e-9) if cost_of_unit > 1e-9 else float('inf')
                vol_to_allocate = min(available_vol, affordable_vol)

            # Allocate if possible
            if vol_to_allocate > 1e-6:
                cost = vol_to_allocate * price
                # Update totals
                total_allocated_volume_year += vol_to_allocate
                total_allocated_cost_year += cost
                # Update tracking dictionaries
                volume_tracker[project_name] -= vol_to_allocate # Decrease remaining volume
                if project_name not in allocated_projects_this_year:
                     allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                # print(f"    Allocated {vol_to_allocate:.2f} to Prio {project['priority']} project '{project_name}' (Price: {price:.2f})") # Optional detailed log


    # --- Stage 2: Allocate Non-Prioritized Projects ---
    print("  Stage 2: Processing non-prioritized projects...")
    limit_met_after_prio = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)

    if limit_met_after_prio:
        print("    Limit already met after prioritized projects.")
    elif non_prio_df.empty:
        print("    No non-prioritized projects to process.")
    else:
        # --- Stage 2.A: Equal Share for Volume Constrained ---
        if constraint_type == "Volume Constrained":
            print("    Applying equal volume share logic (Volume Constrained).")
            # Filter non_prio_df for projects that *still* have volume tracked > 0
            active_non_prio_names = [name for name, vol in volume_tracker.items() if vol > 1e-6 and name in non_prio_df['project name'].values]
            if not active_non_prio_names:
                 print("    No active non-prioritized projects with remaining volume.")
            else:
                # Create a DataFrame view of only active non-prio projects
                active_non_prio_df = non_prio_df[non_prio_df['project name'].isin(active_non_prio_names)].copy()
                # Add current remaining volume from tracker
                active_non_prio_df['current_volume'] = active_non_prio_df['project name'].map(volume_tracker)
                
                types_to_fill = active_non_prio_df['project type'].unique()

                for current_type in types_to_fill:
                    current_limit_vol_start_type = max(0, annual_limit - total_allocated_volume_year)
                    if current_limit_vol_start_type < 1e-6:
                        print("      Global volume limit met, skipping further types.")
                        break 

                    # Get non-prio projects of this type with remaining volume > 0
                    type_non_prio_mask = (active_non_prio_df['project type'] == current_type) & \
                                         (active_non_prio_df['current_volume'] > 1e-6)
                    type_indices = active_non_prio_df.index[type_non_prio_mask]
                    
                    if len(type_indices) == 0: continue 

                    print(f"      Processing type: {current_type} ({len(type_indices)} projects)")
                    # Iterative allocation within the type
                    while True:
                        current_limit_vol = max(0, annual_limit - total_allocated_volume_year)
                        if current_limit_vol < 1e-6: break 

                        # Find projects of this type still active (volume > 1e-6)
                        active_mask_iter = active_non_prio_df.loc[type_indices, 'current_volume'] > 1e-6
                        active_indices_iter = type_indices[active_mask_iter]
                        num_active = len(active_indices_iter)

                        if num_active == 0: break 

                        ideal_share_vol = current_limit_vol / num_active
                        min_remaining_vol_active = active_non_prio_df.loc[active_indices_iter, 'current_volume'].min()
                        vol_per_project_this_iter = min(ideal_share_vol, min_remaining_vol_active)

                        if vol_per_project_this_iter < 1e-9: break 
                        
                        max_alloc_this_iter = vol_per_project_this_iter * num_active
                        if max_alloc_this_iter > current_limit_vol + 1e-9: 
                            vol_per_project_this_iter = current_limit_vol / num_active
                            if vol_per_project_this_iter < 1e-9: break

                        allocated_in_iter = 0
                        for idx in active_indices_iter:
                            project_name = active_non_prio_df.loc[idx, 'project name']
                            price = active_non_prio_df.loc[idx, 'price']
                            proj_type = active_non_prio_df.loc[idx, 'project type']
                            
                            # Ensure we don't allocate more than available for this specific project
                            vol_available_proj = active_non_prio_df.loc[idx, 'current_volume']
                            actual_alloc_proj = min(vol_per_project_this_iter, vol_available_proj)

                            if actual_alloc_proj < 1e-9: continue 

                            # Update tracking and totals
                            active_non_prio_df.loc[idx, 'current_volume'] -= actual_alloc_proj # Update local copy first
                            volume_tracker[project_name] -= actual_alloc_proj # Update master tracker

                            if project_name not in allocated_projects_this_year:
                                allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': proj_type}
                            allocated_projects_this_year[project_name]['volume'] += actual_alloc_proj

                            total_allocated_volume_year += actual_alloc_proj
                            total_allocated_cost_year += actual_alloc_proj * (price if pd.notna(price) else 0)
                            allocated_in_iter += actual_alloc_proj
                            # print(f"      Allocated {actual_alloc_proj:.2f} to NonPrio '{project_name}' (Type: {current_type})") # Optional detailed log

                        if allocated_in_iter < 1e-9: break # Break if no progress made

        # --- Stage 2.B: Sequential Price for Budget Constrained ---
        else: # Budget Constrained
            print("    Applying sequential price-based logic (Budget Constrained).")
            # Use the non_prio_df already sorted by price
            for idx, project in non_prio_df.iterrows():
                # Check limit first
                if total_allocated_cost_year >= annual_limit - 1e-6:
                    print("      Budget limit reached.")
                    break

                project_name = project['project name']
                available_vol = volume_tracker.get(project_name, 0) # Get current available volume
                price = project['price']
                cost_of_unit = price if price > 0 else 0

                if available_vol < 1e-6: continue # Skip if already fully allocated

                vol_to_allocate = 0.0
                remaining_limit = max(0, annual_limit - total_allocated_cost_year)
                affordable_vol = remaining_limit / (cost_of_unit + 1e-9) if cost_of_unit > 1e-9 else float('inf')
                vol_to_allocate = min(available_vol, affordable_vol)

                if vol_to_allocate > 1e-6:
                    cost = vol_to_allocate * price
                    # Update totals
                    total_allocated_volume_year += vol_to_allocate
                    total_allocated_cost_year += cost
                    # Update tracking dictionaries
                    volume_tracker[project_name] -= vol_to_allocate
                    if project_name not in allocated_projects_this_year:
                         allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                    allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                    # print(f"    Allocated {vol_to_allocate:.2f} to NonPrio '{project_name}' (Price: {price:.2f})") # Optional detailed log


        # --- Final Step for Year ---
    
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
        summary_list = []
        for year in selected_years:
            year_data = portfolio_df[portfolio_df['year'] == year]
            total_volume = year_data['volume'].sum()
            total_cost = year_data['cost'].sum()
            avg_price = total_cost / (total_volume + 1e-9)
            volume_by_type = year_data.groupby('type')['volume'].sum()
            summary_entry = {'Year': year, 'Total Volume (tonnes)': total_volume, 'Total Cost (€)': total_cost, 'Average Price (€/tonne)': avg_price, 'Target Constraint': annual_constraints.get(year, 0)}
            # Add volumes for all potentially selected types, defaulting to 0 if not in this year's allocation
            for proj_type in (global_removal_types + ([global_reduction_type] if global_reduction_type else [])):
                  summary_entry[f'Volume {proj_type.capitalize()}'] = volume_by_type.get(proj_type, 0)
            summary_list.append(summary_entry)
        summary_df = pd.DataFrame(summary_list)

        st.subheader("Portfolio Composition & Price Over Time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Define UPDATED consistent colors
        color_map = {
            'technical removal': '#64B5F6', # Blueish (Material Blue Lighten 2)
            'natural removal': '#81C784', # Greenish (Material Green Lighten 2)
            'reduction': '#B0BEC5', # Greyish (Material Blue Grey Lighten 1)
            # Add more types and colors if needed
        }
        default_color = '#D3D3D3' # Light Grey for unknown types

        # Plot types present in the actual allocation
        plot_types = sorted([t for t in all_types_in_portfolio if t in color_map])

        for type_name in plot_types:
            type_volume_col = f'Volume {type_name.capitalize()}'
            if type_volume_col in summary_df.columns:
                type_volume = summary_df[type_volume_col]
                fig.add_trace(go.Bar(x=summary_df['Year'], y=type_volume, name=type_name.capitalize(), marker_color=color_map.get(type_name, default_color)), secondary_y=False)

        fig.add_trace(go.Scatter(x=summary_df['Year'], y=summary_df['Average Price (€/tonne)'], name='Avg Price (€/tonne)', marker=dict(symbol='circle'), line=dict(color='#546E7A')), secondary_y=True) # Darker Blue Grey for price line

        fig.update_layout(xaxis_title='Year', yaxis_title='Volume (tonnes)', yaxis2_title='Average Price (€/tonne)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white", yaxis=dict(rangemode='tozero'), yaxis2=dict(rangemode='tozero'))
        st.plotly_chart(fig, use_container_width=True)

        # Final Year Removal %
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
        else:
             st.info(f"No data allocated for the final year ({end_year}).")

        st.subheader("Yearly Summary")
        summary_display_df = summary_df.copy()
        # Standardize column names before display
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

        # Add type columns to display if they exist in the summary
        for proj_type in plot_types:
             col_name = f'Volume {proj_type.capitalize()}'
             if col_name in summary_display_df.columns:
                  display_cols.append(col_name)
                  summary_display_df[col_name] = summary_display_df[col_name].map('{:,.0f}'.format) # Format volume

        # Format numeric columns
        for col in ['Target Budget (€)', 'Achieved Cost (€)', 'Avg Price (€/tonne)']:
             if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('{:,.2f}'.format)
        for col in ['Target Volume', 'Achieved Volume']:
              if col in summary_display_df.columns: summary_display_df[col] = summary_display_df[col].map('{:,.0f}'.format)


        st.dataframe(summary_display_df[display_cols].set_index('Year'))


        # --- Convert Portfolio Dictionary to DataFrame (Summing volumes across all years) ---
        records = []
        for year, projects in portfolio.items():
            for name, info in projects.items():
                if info['volume'] > 0:
                    records.append({
                        'year': str(year),
                        'type': info['type'],
                        'project': name,
                        'volume': info['volume']
                    })
        
        df = pd.DataFrame(records)
        
        # --- Aggregate the Data (sum volumes across all years for each project) ---
        df_summed = df.groupby(['type', 'project']).agg({'volume': 'sum'}).reset_index()
        
        # --- Sunburst Plot for Total Allocation Across All Years ---
        fig = px.sunburst(
            df_summed,
            path=['type', 'project'],  # Hierarchical structure: Type -> Project
            values='volume',           # Size based on the total volume across all years
            color='type',              # Color by type (Technical, Natural, Reduction)
            color_discrete_map={
                'technical removal': 'royalblue',
                'natural removal': 'forestgreen',
                'reduction': 'orange'
            },
            title="Total Carbon Credit Allocation Across All Years"
        )
        
        fig.update_traces(textinfo='label+value+percent entry')
        
        # --- Display the Chart in Streamlit ---
        st.subheader("📊 Total Allocation Overview (Across All Years Combined)")
        st.plotly_chart(fig, use_container_width=True)

    
        # Raw Allocation Data
        st.subheader("Detailed Allocation Data")
        if st.checkbox("Show raw project allocations by year", key="show_raw"):
             display_portfolio_df = portfolio_df.copy()
             display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.2f}'.format)
             display_portfolio_df['price'] = display_portfolio_df['price'].map('{:,.2f}'.format)
             display_portfolio_df['cost'] = display_portfolio_df['cost'].map('{:,.2f}'.format)
             st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']))

    elif df_upload:
         st.warning("Please select at least one project in Step 2 to build the portfolio.")
