import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np # Added for potential NaN checks if needed

# --- Streamlit App Configuration ---
st.markdown("""
    <style>
    body {
        background-color: #D9E1C8;
    }
    .stApp {
        background-color: #D9E1C8;
    }
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
    start_year = 2025 # Define start year explicitly
    end_year = start_year + years - 1
    selected_years = list(range(start_year, end_year + 1))

    overview = data.copy()
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in overview.columns]

    # Ensure 'priority' column exists, fill with a default if not (e.g., 50 or median)
    if 'priority' not in overview.columns:
        overview['priority'] = 50 # Assign a neutral default priority
        st.info("No 'priority' column found in CSV. Assigning a default priority of 50 to all projects.")
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
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']))
    else:
        overview_display_cols = ['project name', 'project type']
        if 'description' in overview.columns:
            overview_display_cols.append('description')
        st.dataframe(overview[overview_display_cols].drop_duplicates(subset=['project name']))
        st.warning(f"No price columns found for years {start_year}-{end_year} (e.g., 'price {start_year}'). Cannot calculate average price.")


    # --- Portfolio Settings Section ---
    st.subheader("Step 1: Define Portfolio Settings")
    constraint_type = st.radio("Select constraint type:", ["Volume Constrained", "Budget Constrained"], key="constraint_type")
    annual_constraints = {}

    col1, col2 = st.columns(2)
    with col1:
        if constraint_type == "Volume Constrained":
            st.markdown("**Enter annual purchase volumes (tonnes):**")
            global_volume = st.number_input("Set volume for all years (optional):", min_value=0, step=100, value=0)
            for year in selected_years:
                default_val = global_volume if global_volume > 0 else 1000
                annual_constraints[year] = st.number_input(f"Volume {year}:", min_value=0, step=100, value=default_val, key=f"vol_{year}")
        else:
            st.markdown("**Enter annual budget (€):**")
            global_budget = st.number_input("Set budget for all years (optional, €):", min_value=0, step=1000, value=0)
            for year in selected_years:
                default_val = global_budget if global_budget > 0 else 10000
                annual_constraints[year] = st.number_input(f"Budget {year} (€):", min_value=0, step=1000, value=default_val, key=f"bud_{year}")

    with col2:
        st.markdown("**Define Portfolio Strategy:**")
        removal_target = st.slider(f"Target Removal % by {end_year}", 0, 100, 80, key="removal_target") / 100
        transition_speed = st.slider("Transition Speed (1=Slow, 10=Fast)", 1, 10, 5, key="transition_speed")
        removal_preference = st.slider("Removal Preference (1=Natural, 10=Technical)", 1, 10, 5, key="removal_preference")

    # --- Project Selection and Prioritization Section ---
    st.subheader("Step 2: Select and Prioritize Projects")
    project_names = data['project name'].unique().tolist()
    selected_projects = st.multiselect("Select projects to include:", project_names, default=project_names, key="select_proj") # Default to all
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

        # Identify available project types
        selected_types = selected_df['project type'].unique().tolist()
        removal_types = [t for t in selected_types if t in ['technical removal', 'natural removal']]
        reduction_type = 'reduction' if 'reduction' in selected_types else None
        has_removals = bool(removal_types)
        has_reductions = bool(reduction_type)

        # Check for edge cases and inform user
        only_removals = has_removals and not has_reductions
        only_reductions = not has_removals and has_reductions
        if only_removals:
            st.info("Only removal projects selected. Transition targets apply only to removals.")
        if only_reductions:
            st.info("Only reduction projects selected. Transition targets and removal preference ignored.")

        # --- Allocation Logic ---
        portfolio = {year: {} for year in selected_years} # Stores { project_name: { volume: v, price: p, type: t } }
        allocation_warnings = []

        for year_idx, year in enumerate(selected_years):
            year_str = str(year)
            volume_col = f"available volume {year_str}"
            price_col = f"price {year_str}"

            # Check if essential columns exist for the current year
            if volume_col not in selected_df.columns or price_col not in selected_df.columns:
                allocation_warnings.append(f"Warning for {year}: Missing '{volume_col}' or '{price_col}' column. Skipping allocation for this year.")
                portfolio[year] = {} # Ensure entry exists even if skipped
                continue # Skip allocation for this year

            # Prepare DataFrame for this year's allocation
            year_df = selected_df[['project name', 'project type', 'priority', volume_col, price_col]].copy()
            year_df.rename(columns={volume_col: 'volume', price_col: 'price'}, inplace=True)

            # Convert volume and price to numeric, handle errors/NaNs
            year_df['volume'] = pd.to_numeric(year_df['volume'], errors='coerce').fillna(0)
            year_df['price'] = pd.to_numeric(year_df['price'], errors='coerce') # Keep NaN price for now

            # Filter out projects with zero volume or invalid price for the year
            # Keep projects with NaN price if budget constrained (might be free?) - No, filter NaNs always.
            year_df = year_df[(year_df['volume'] > 0) & pd.notna(year_df['price'])]

            if year_df.empty:
                allocation_warnings.append(f"Warning for {year}: No projects with available volume and valid price found.")
                portfolio[year] = {}
                continue

            # --- Year Specific Calculations ---
            annual_limit = annual_constraints.get(year, 0)
            if annual_limit <= 0: # Skip year if constraint is zero or negative
                 allocation_warnings.append(f"Note for {year}: Annual constraint is zero or less. No allocation performed.")
                 portfolio[year] = {}
                 continue

            total_allocated_volume_year = 0.0
            total_allocated_cost_year = 0.0
            allocated_projects_this_year = {} # { name: {'volume': v, 'price': p, 'type': t} }

            # Calculate target split for removals/reductions based on transition
            year_fraction = (year_idx + 1) / years
            current_removal_target_fraction = removal_target * (year_fraction ** (0.5 + 0.1 * transition_speed))

            # --- Determine if only removals are selected FOR THIS YEAR'S available projects ---
            current_year_types = year_df['project type'].unique().tolist()
            current_has_removals = any(t in ['technical removal', 'natural removal'] for t in current_year_types)
            current_has_reductions = 'reduction' in current_year_types
            current_only_removals = current_has_removals and not current_has_reductions
            current_only_reductions = not current_has_removals and current_has_reductions
            # ---

            if current_only_reductions:
                 current_removal_target_fraction = 0.0
            elif current_only_removals:
                 current_removal_target_fraction = 1.0 # Target is 100% removals

            target_removal_alloc = annual_limit * current_removal_target_fraction
            target_reduction_alloc = annual_limit * (1.0 - current_removal_target_fraction)

            # Sort projects: High priority first, then low price
            year_df.sort_values(by=['priority', 'price'], ascending=[False, True], inplace=True)

            # Create lists of projects by type for easier iteration
            removals_df = year_df[year_df['project type'].isin(removal_types)].copy()
            reductions_df = year_df[year_df['project type'] == reduction_type].copy() if reduction_type else pd.DataFrame()

            # --- PHASE 1: Targeted Allocation ---
            # Allocate based on preferences and calculated targets, but don't exceed annual_limit

            # 1.a Allocate Removals (respecting preference)
            if current_has_removals and not removals_df.empty:
                removal_pref_factor = removal_preference / 10.0
                # These preference targets are mainly for the mixed scenario
                natural_pref_target = target_removal_alloc * (1.0 - removal_pref_factor)
                technical_pref_target = target_removal_alloc * removal_pref_factor

                # Determine allocation order based on preference slider
                removal_order_types = ['natural removal', 'technical removal']
                if removal_preference > 5:
                    removal_order_types.reverse() # Prioritize technical if slider > 5

                current_removal_allocated_value = 0.0 # Track value (volume or cost) allocated towards the removal target specifically

                for r_type in removal_order_types:
                    if r_type not in removals_df['project type'].unique(): continue

                    type_df = removals_df[removals_df['project type'] == r_type].copy() # Use copy to avoid SettingWithCopyWarning if modifying later

                    for idx, project in type_df.iterrows():
                        # Check if overall limit is already reached before processing project
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6):
                            break # Stop allocating this type if overall limit hit

                        project_name = project['project name']
                        # Use .loc to get potentially updated volume if same project appears multiple times (shouldn't happen with unique names)
                        available_vol = year_df.loc[idx, 'volume'] # Get current available volume
                        price = project['price']

                        if available_vol < 1e-6: continue # Skip if volume already exhausted

                        # Calculate how much *can* be allocated based on constraints
                        vol_to_allocate = 0.0

                        if constraint_type == "Volume Constrained":
                            remaining_overall_limit = max(0, annual_limit - total_allocated_volume_year)
                            # **MODIFICATION for only_removals:** Don't limit by sub-type target if only removals exist
                            if current_only_removals:
                                vol_to_allocate = min(available_vol, remaining_overall_limit)
                            else:
                                # Original logic for mixed portfolio (limit by sub-target AND overall)
                                remaining_type_target_vol = max(0, target_removal_alloc - current_removal_allocated_value) # How much more needed for *overall* removal target
                                vol_to_allocate = min(available_vol, remaining_type_target_vol, remaining_overall_limit)
                                # Note: We previously used r_target (type-specific target), now using overall removal target here too for simplicity.

                        else: # Budget Constrained
                            cost_of_unit = price if price > 0 else 0
                            remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                            affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9)

                            # **MODIFICATION for only_removals:** Don't limit by sub-type target budget if only removals exist
                            if current_only_removals:
                                vol_to_allocate = min(available_vol, affordable_vol_overall)
                            else:
                                # Original logic for mixed portfolio
                                remaining_type_target_budget = max(0, target_removal_alloc - total_allocated_cost_year) # How much budget left for *all* removals
                                affordable_vol_type = remaining_type_target_budget / (cost_of_unit + 1e-9)
                                vol_to_allocate = min(available_vol, affordable_vol_type, affordable_vol_overall)

                        # Clean up potential floating point issues for very small amounts
                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0

                        # Allocate if possible
                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            # Update totals
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost
                            # Update value allocated specifically to removals
                            current_removal_allocated_value += vol_to_allocate if constraint_type == "Volume Constrained" else cost

                            # Store allocation (update if project already partially allocated - shouldn't happen with this loop structure)
                            if project_name not in allocated_projects_this_year:
                                allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate

                            # Reduce available volume for this project IN THE MAIN year_df (for Phase 2)
                            year_df.loc[idx, 'volume'] -= vol_to_allocate

                    # Check again after finishing a type if overall limit hit
                    if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                       (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6):
                        break # Stop allocating more removal types if overall limit hit

            # 1.b Allocate Reductions (Only if reductions exist and limit not met)
            if current_has_reductions and not reductions_df.empty:
                 # Check if overall limit is already reached before starting reductions
                 if not ((constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                         (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)):

                    for idx, project in reductions_df.iterrows():
                         # Check overall limit before each project
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6):
                            break

                        project_name = project['project name']
                        available_vol = year_df.loc[idx, 'volume'] # Get current remaining volume
                        price = project['price']

                        if available_vol < 1e-6: continue

                        vol_to_allocate = 0.0

                        if constraint_type == "Volume Constrained":
                             remaining_overall_limit = max(0, annual_limit - total_allocated_volume_year)
                             # In Phase 1 for reductions, we are limited by the remaining *overall* limit AND the reduction target
                             remaining_reduction_target_vol = max(0, target_reduction_alloc - (total_allocated_volume_year - current_removal_allocated_value)) # Volume target for reductions
                             vol_to_allocate = min(available_vol, remaining_overall_limit, remaining_reduction_target_vol)

                        else: # Budget Constrained
                             cost_of_unit = price if price > 0 else 0
                             remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                             affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9)
                             # Limited by remaining overall budget and reduction target budget
                             remaining_reduction_target_budget = max(0, target_reduction_alloc - (total_allocated_cost_year - current_removal_allocated_value))# Budget target for reductions
                             affordable_vol_reduction_target = remaining_reduction_target_budget / (cost_of_unit + 1e-9)
                             vol_to_allocate = min(available_vol, affordable_vol_overall, affordable_vol_reduction_target)

                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0

                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost

                            if project_name not in allocated_projects_this_year:
                                allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            year_df.loc[idx, 'volume'] -= vol_to_allocate


            # --- PHASE 2: Gap Filling ---
            # If annual_limit not met, fill with best remaining projects regardless of type target

            limit_met = (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                        (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6)

            if not limit_met and annual_limit > 0:
                # Get remaining projects with volume > 0, sort again by priority/price
                remaining_projects_df = year_df[year_df['volume'] > 1e-6].sort_values(by=['priority', 'price'], ascending=[False, True])

                if not remaining_projects_df.empty:
                    # allocation_warnings.append(f"Debug for {year}: Entering Phase 2. Limit not met ({total_allocated_volume_year if constraint_type=='Volume Constrained' else total_allocated_cost_year} / {annual_limit}).") # Optional debug
                    for idx, project in remaining_projects_df.iterrows():
                        # Check limit before each project
                        if (constraint_type == "Volume Constrained" and total_allocated_volume_year >= annual_limit - 1e-6) or \
                           (constraint_type == "Budget Constrained" and total_allocated_cost_year >= annual_limit - 1e-6):
                            break

                        project_name = project['project name']
                        available_vol = project['volume'] # Remaining volume from year_df
                        price = project['price']

                        vol_to_allocate = 0.0

                        if constraint_type == "Volume Constrained":
                            remaining_overall_limit = max(0, annual_limit - total_allocated_volume_year)
                            vol_to_allocate = min(available_vol, remaining_overall_limit)
                        else: # Budget Constrained
                            cost_of_unit = price if price > 0 else 0
                            remaining_overall_budget = max(0, annual_limit - total_allocated_cost_year)
                            affordable_vol_overall = remaining_overall_budget / (cost_of_unit + 1e-9)
                            vol_to_allocate = min(available_vol, affordable_vol_overall)

                        if vol_to_allocate < 1e-6 : vol_to_allocate = 0.0

                        if vol_to_allocate > 0:
                            cost = vol_to_allocate * price
                            total_allocated_volume_year += vol_to_allocate
                            total_allocated_cost_year += cost

                            # Add or update allocation for this project
                            if project_name not in allocated_projects_this_year:
                                allocated_projects_this_year[project_name] = {'volume': 0.0, 'price': price, 'type': project['project type']}
                            allocated_projects_this_year[project_name]['volume'] += vol_to_allocate
                            # We don't strictly need to decrement year_df volume in phase 2, but good practice:
                            # year_df.loc[idx, 'volume'] -= vol_to_allocate # careful if project listed multiple times

            # Store final allocation for the year
            portfolio[year] = allocated_projects_this_year

            # Add warning if constraint still not met after both phases (due to lack of available projects)
            final_limit_met = (constraint_type == "Volume Constrained" and abs(total_allocated_volume_year - annual_limit) < 1e-6) or \
                              (constraint_type == "Budget Constrained" and abs(total_allocated_cost_year - annual_limit) < 1e-6)
            # Also check if we allocated *something* if the limit was > 0
            limit_partially_met = (total_allocated_volume_year > 1e-6) if constraint_type == "Volume Constrained" else (total_allocated_cost_year > 1e-6)


            if not final_limit_met and annual_limit > 0 and not limit_partially_met:
                 # Special case: No projects allocated at all, likely due to filtering or 0 available
                  allocation_warnings.append(f"Warning for {year}: Could not allocate *any* volume/budget towards the target of {annual_limit:.2f}. Check project availability and prices for this year.")
            elif not final_limit_met and annual_limit > 0:
                 # Standard case: Allocated some, but couldn't meet the full target
                 if constraint_type == "Volume Constrained":
                      allocation_warnings.append(f"Warning for {year}: Could only allocate {total_allocated_volume_year:.2f} tonnes out of the target {annual_limit:.0f} tonnes due to insufficient project availability.")
                 else:
                      allocation_warnings.append(f"Warning for {year}: Could only spend €{total_allocated_cost_year:.2f} out of the target budget €{annual_limit:.2f} due to insufficient affordable project volume.")

        # --- Display Warnings ---
        if allocation_warnings:
            st.warning("Allocation Notes & Warnings:")
            for warning in allocation_warnings:
                st.markdown(f"- {warning}")

        # --- Portfolio Analysis and Visualization ---
        all_types_in_portfolio = set()
        portfolio_data_list = []
        for year, projects in portfolio.items():
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
            
            summary_entry = {
                'Year': year,
                'Total Volume (tonnes)': total_volume,
                'Total Cost (€)': total_cost,
                'Average Price (€/tonne)': avg_price,
                'Target Constraint': annual_constraints.get(year, 0)
            }
            for proj_type in all_types_in_portfolio:
                 summary_entry[f'Volume {proj_type.capitalize()}'] = volume_by_type.get(proj_type, 0)
            
            summary_list.append(summary_entry)

        summary_df = pd.DataFrame(summary_list)

        st.subheader("Portfolio Composition & Price Over Time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Define consistent colors
        color_map = {
            'technical removal': '#8BC34A', # Darker Green
            'natural removal': '#AED581', # Medium Green
            'reduction': '#C5E1A5', # Lighter Green
            # Add more types and colors if needed
        }
        default_color = '#D3D3D3' # Grey for unknown types

        plot_types = sorted(list(all_types_in_portfolio)) # Sort for consistent legend order

        for type_name in plot_types:
            type_volume = summary_df[f'Volume {type_name.capitalize()}']
            fig.add_trace(go.Bar(
                x=summary_df['Year'],
                y=type_volume,
                name=type_name.capitalize(),
                marker_color=color_map.get(type_name, default_color)
                ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=summary_df['Year'],
            y=summary_df['Average Price (€/tonne)'],
            name='Average Price (€/tonne)',
            marker=dict(symbol='circle'),
            line=dict(color='#558B2F') # Dark olive
            ), secondary_y=True)

        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Volume (tonnes)',
            yaxis2_title='Average Price (€/tonne)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            barmode='stack',
            template="plotly_white",
            yaxis=dict(rangemode='tozero'), # Ensure volume axis starts at 0
            yaxis2=dict(rangemode='tozero') # Ensure price axis starts at 0
        )
        st.plotly_chart(fig, use_container_width=True)

        # Final Year Removal %
        final_year_summary = summary_df[summary_df['Year'] == end_year].iloc[0]
        final_tech = final_year_summary.get(f'Volume Technical removal', 0)
        final_nat = final_year_summary.get(f'Volume Natural removal', 0)
        final_total_removal = final_tech + final_nat
        final_total = final_year_summary['Total Volume (tonnes)']
        achieved_removal_perc = (final_total_removal / (final_total + 1e-9)) * 100

        if has_removals:
             st.metric(label=f"Achieved Removal % in {end_year}", value=f"{achieved_removal_perc:.2f}%")
        elif not selected_projects:
             st.info("No projects selected for the portfolio.")
        else:
             st.info("No removal projects selected.")


        st.subheader("Yearly Summary")
        summary_display_df = summary_df.copy()
        if constraint_type == "Volume Constrained":
            summary_display_df.rename(columns={'Target Constraint': 'Target Volume (tonnes)', 'Total Volume (tonnes)': 'Achieved Volume (tonnes)'}, inplace=True)
            display_cols = ['Year', 'Target Volume (tonnes)', 'Achieved Volume (tonnes)', 'Total Cost (€)', 'Average Price (€/tonne)']
        else:
             summary_display_df.rename(columns={'Target Constraint': 'Target Budget (€)', 'Total Cost (€)': 'Achieved Cost (€)'}, inplace=True)
             display_cols = ['Year', 'Target Budget (€)', 'Achieved Cost (€)', 'Total Volume (tonnes)', 'Average Price (€/tonne)']

        # Add type columns to display
        for proj_type in plot_types:
             display_cols.append(f'Volume {proj_type.capitalize()}')

        # Format for display
        summary_display_df['Achieved Cost (€)'] = summary_display_df['Achieved Cost (€)'].map('{:,.2f}'.format) if 'Achieved Cost (€)' in summary_display_df else None
        summary_display_df['Total Cost (€)'] = summary_display_df['Total Cost (€)'].map('{:,.2f}'.format) if 'Total Cost (€)' in summary_display_df else None
        summary_display_df['Target Budget (€)'] = summary_display_df['Target Budget (€)'].map('{:,.2f}'.format) if 'Target Budget (€)' in summary_display_df else None
        summary_display_df['Average Price (€/tonne)'] = summary_display_df['Average Price (€/tonne)'].map('{:,.2f}'.format)
        for col in display_cols:
            if 'Volume' in col :
                 summary_display_df[col] = summary_display_df[col].map('{:,.0f}'.format)

        st.dataframe(summary_display_df[display_cols].set_index('Year'))


        # Raw Allocation Data
        st.subheader("Detailed Allocation Data")
        if st.checkbox("Show raw project allocations by year"):
             display_portfolio_df = portfolio_df.copy()
             display_portfolio_df['volume'] = display_portfolio_df['volume'].map('{:,.2f}'.format)
             display_portfolio_df['price'] = display_portfolio_df['price'].map('{:,.2f}'.format)
             display_portfolio_df['cost'] = display_portfolio_df['cost'].map('{:,.2f}'.format)
             st.dataframe(display_portfolio_df[['year', 'project name', 'type', 'volume', 'price', 'cost']].sort_values(by=['year', 'project name']))

    elif df_upload: # df_upload exists but selected_projects is empty
         st.warning("Please select at least one project in Step 2 to build the portfolio.")

# Optional: Add a footer or instructions if needed
# st.markdown("---")
# st.caption("Upload a CSV with columns like 'project name', 'project type', 'priority', 'available volume YEAR', 'price YEAR'.")
