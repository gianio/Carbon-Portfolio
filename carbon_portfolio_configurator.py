import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
df = st.file_uploader("Upload project CSV", type="csv")

# --- Main Logic (Conditional on CSV Upload) ---
if df:
    data = pd.read_csv(df)

    # --- Project Overview Section ---
    st.subheader("Project Overview")
    years = st.number_input("How many years should the portfolio span?", min_value=1, max_value=20, value=6, step=1)
    end_year = 2025 + years - 1
    selected_years = list(range(2025, end_year + 1))
    overview = data.copy()
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in data.columns]
    if price_cols:
        overview['avg_price'] = overview[price_cols].mean(axis=1)
        if 'description' in overview.columns:
            st.dataframe(overview[['project name', 'project type', 'description', 'avg_price']].drop_duplicates())
        else:
            st.dataframe(overview[['project name', 'project type', 'avg_price']].drop_duplicates())
    else:
        if 'description' in overview.columns:
            st.dataframe(overview[['project name', 'project type', 'description']].drop_duplicates())
        else:
            st.dataframe(overview[['project name', 'project type']].drop_duplicates())

    # --- Portfolio Settings Section ---
    st.subheader("Step 1: Define Portfolio Settings")
    constraint_type = st.radio("Are you budget constrained or volume constrained?", ["Volume Constrained", "Budget Constrained"])
    annual_constraints = {}
    if constraint_type == "Volume Constrained":
        st.markdown("**Enter annual purchase volumes (in tonnes):**")
        for year in selected_years:
            annual_constraints[year] = st.number_input(f"Annual purchase volume for {year}:", min_value=0, step=100, value=1000)
    else:
        st.markdown("**Enter annual budget (in €):**")
        for year in selected_years:
            annual_constraints[year] = st.number_input(f"Annual budget for {year} (€):", min_value=0, step=1000, value=10000)

    removal_target = st.slider(f"Target total removal % for year {end_year}", 0, 100, 80) / 100
    transition_speed = st.slider("Transition Speed (1 = Slow, 10 = Fast)", 1, 10, 5)
    removal_preference = st.slider("Removal Preference (1 = Natural, 10 = Technical)", 1, 10, 5)

    # --- Project Selection and Prioritization Section ---
    st.subheader("Step 2: Select and Prioritize Projects")
    project_names = data['project name'].unique().tolist()
    selected_projects = st.multiselect("Select projects to include in the portfolio:", project_names)
    favorite_projects = st.multiselect("Select your favorite projects (will get a +10% priority boost):", selected_projects)

    if selected_projects:
        selected_df = data[data['project name'].isin(selected_projects)].copy()
        selected_types = selected_df['project type'].unique().tolist()
        only_removals = all(t in ['technical removal', 'natural removal'] for t in selected_types)
        only_reductions = all(t == 'reduction' for t in selected_types)

        if only_removals:
            st.info("Only removal projects are selected. Transition and removal preference will be ignored.")
        elif only_reductions:
            st.info("Only reduction projects are selected. Transition and removal preference will be ignored.")

        if favorite_projects:
            selected_df['priority'] = selected_df.apply(
                lambda row: row['priority'] + 10 if row['project name'] in favorite_projects and pd.notna(row['priority']) and row['priority'] < 100 else row['priority'],
                axis=1
            )
            selected_df['priority'] = selected_df['priority'].clip(upper=100)

        removal_types = ['technical removal', 'natural removal']
        reduction_type = ['reduction']
        all_types = selected_types

        project_types = {t: selected_df[selected_df['project type'] == t] for t in all_types}

        portfolio = {year: {} for year in selected_years}
        broken_rules = []

        for year_idx, year in enumerate(selected_years):
            year_str = f"{year}"
            annual_limit = annual_constraints.get(year, 0)
            allocated_projects = {}
            total_allocated_volume = 0
            total_allocated_cost = 0

            removal_preference_factor = removal_preference / 10
            natural_preference_factor = 1 - removal_preference_factor

            target_removal_volume_year = annual_limit * (removal_target * ((year - 2024) / years)**(0.5 + 0.1 * transition_speed)) if 'technical removal' in all_types or 'natural removal' in all_types else 0
            target_reduction_volume_year = annual_limit * (1 - (removal_target * ((year - 2024) / years)**(0.5 + 0.1 * transition_speed))) if 'reduction' in all_types and ('technical removal' in all_types or 'natural removal' in all_types) else annual_limit if only_reductions and constraint_type == "Volume Constrained" else 0

            target_budget_removal_year = annual_limit * (removal_target * ((year - 2024) / years)**(0.5 + 0.1 * transition_speed)) if 'technical removal' in all_types or 'natural removal' in all_types else 0
            target_budget_reduction_year = annual_limit * (1 - (removal_target * ((year - 2024) / years)**(0.5 + 0.1 * transition_speed))) if 'reduction' in all_types and ('technical removal' in all_types or 'natural removal' in all_types) else annual_limit if only_reductions and constraint_type == "Budget Constrained" else 0

            allocated_removal_volume_year = 0
            allocated_reduction_volume_year = 0
            allocated_cost_removal_year = 0
            allocated_cost_reduction_year = 0
            met_removal_target = False

            # Allocate Removals with Preference
            for removal_type, preference in [('natural removal', natural_preference_factor), ('technical removal', removal_preference_factor)]:
                available_projects_type = project_types.get(removal_type, pd.DataFrame()).copy()
                volume_col = f"available volume {year_str}"
                price_col = f"price {year_str}"

                if not available_projects_type.empty and volume_col in available_projects_type.columns and price_col in available_projects_type.columns:
                    available_projects_type = available_projects_type[pd.notna(available_projects_type[volume_col]) & (available_projects_type[volume_col] > 0)].sort_values(by=['priority' if 'priority' in available_projects_type.columns else price_col, price_col], ascending=[False if 'priority' in available_projects_type.columns else True, True])

                    if constraint_type == "Volume Constrained" and not met_removal_target:
                        target_volume_type = target_removal_volume_year * preference
                        allocated_volume_type = 0
                        for _, project in available_projects_type.iterrows():
                            can_allocate = min(target_volume_type - allocated_volume_type, project.get(volume_col, 0))
                            if can_allocate > 0:
                                allocated_projects[project['project name']] = {'volume': can_allocate, 'price': project.get(price_col, 0), 'type': project['project type']}
                                allocated_removal_volume_year += can_allocate
                                allocated_volume_type += can_allocate
                                total_allocated_volume += can_allocate
                                if allocated_removal_volume_year >= target_removal_volume_year:
                                    met_removal_target = True
                                    break
                            if allocated_volume_type >= target_volume_type:
                                break

                    elif constraint_type == "Budget Constrained" and not met_removal_target:
                        target_budget_type = target_budget_removal_year * preference
                        allocated_cost_type = 0
                        for _, project in available_projects_type.iterrows():
                            affordable_volume = (target_budget_type - allocated_cost_type) / (project.get(price_col, 0) + 1e-9)
                            can_allocate = min(affordable_volume, project.get(volume_col, 0))
                            cost = can_allocate * project.get(price_col, 0)
                            if cost > 0:
                                allocated_projects[project['project name']] = {'volume': can_allocate, 'price': project.get(price_col, 0), 'type': project['project type']}
                                allocated_cost_removal_year += cost
                                total_allocated_volume += can_allocate
                                total_allocated_cost += cost
                                allocated_cost_type += cost
                                if allocated_cost_removal_year >= target_budget_removal_year:
                                    met_removal_target = True
                                    break
                            if allocated_cost_type >= target_budget_type:
                                break

            # Check if removal target was met, if not, try to fulfill with the other type
            if not met_removal_target and ('technical removal' in all_types and 'natural removal' in all_types):
                remaining_removal_target_volume = target_removal_volume_year - allocated_removal_volume_year if constraint_type == "Volume Constrained" else 0
                remaining_removal_target_budget = target_budget_removal_year - allocated_cost_removal_year if constraint_type == "Budget Constrained" else 0

                alternative_removal_types = [t for t in removal_types if t not in [('natural removal', natural_preference_factor)[0] if natural_preference_factor >= removal_preference_factor else ('technical removal', removal_preference_factor)[0]]]
                if alternative_removal_types:
                    alternative_type = alternative_removal_types[0]
                    available_alternative = project_types.get(alternative_type, pd.DataFrame()).copy()
                    if not available_alternative.empty and volume_col in available_alternative.columns and price_col in available_alternative.columns:
                        available_alternative = available_alternative[pd.notna(available_alternative[volume_col]) & (available_alternative[volume_col] > 0)].sort_values(by=['priority' if 'priority' in available_alternative.columns else price_col, price_col], ascending=[False if 'priority' in available_alternative.columns else True, True])
                        if constraint_type == "Volume Constrained":
                            for _, project in available_alternative.iterrows():
                                can_allocate = min(remaining_removal_target_volume, project.get(volume_col, 0))
                                if can_allocate > 0:
                                    allocated_projects[project['project name']] = {'volume': can_allocate, 'price': project.get(price_col, 0), 'type': project['project type']}
                                    allocated_removal_volume_year += can_allocate
                                    total_allocated_volume += can_allocate
                                    if allocated_removal_volume_year >= target_removal_volume_year:
                                        break
                        elif constraint_type == "Budget Constrained":
                            for _, project in available_alternative.iterrows():
                                affordable_volume = (remaining_removal_target_budget) / (project.get(price_col, 0) + 1e-9)
                                can_allocate = min(affordable_volume, project.get(volume_col, 0))
                                cost = can_allocate * project.get(price_col, 0)
                                if cost > 0:
                                    allocated_projects[project['project name']] = {'volume': can_allocate, 'price': project.get(price_col, 0), 'type': project['project type']}
                                    allocated_cost_removal_year += cost
                                    total_allocated_volume += can_allocate
                                    total_allocated_cost += cost
                                    if allocated_cost_removal_year >= target_budget_removal_year:
                                        break

            if allocated_removal_volume_year < target_removal_volume_year and ('technical removal' in all_types or 'natural removal' in all_types):
                st.warning(f"Not enough removal volume available in {year} to meet the target of {target_removal_volume_year:.2f} tonnes. Achieved: {allocated_removal_volume_year:.2f} tonnes.")
            elif allocated_cost_removal_year < target_budget_removal_year and ('technical removal' in all_types or 'natural removal' in all_types):
                st.warning(f"Not enough removal budget available in {year} (€{target_budget_removal_year:.2f}) to potentially meet the removal target. Spent: €{allocated_cost_removal_year:.2f}.")

            # Allocate Reductions
            available_reductions = project_types.get('reduction', pd.DataFrame()).copy()
            if not available_reductions.empty and volume_col in available_reductions.columns and price_col in available_reductions.columns:
                available_reductions = available_reductions[pd.notna(available_reductions[volume_col]) & (available_reductions[volume_col] > 0)].sort_values(by=['priority' if 'priority' in available_reductions.columns else price_col, price_col], ascending=[False if 'priority' in available_reductions.columns else True, True])

                if constraint_type == "Volume Constrained":
                    for _, project in available_reductions.iterrows():
                        can_allocate = min(target_reduction_volume_year - allocated_reduction_volume_year, project.get(volume_col, 0))
                        if can_allocate > 0:
                            allocated_projects[project['project name']] = {'volume': can_allocate, 'price': project.get(price_col, 0), 'type': 'reduction'}
                            allocated_reduction_volume_year += can_allocate
                            total_allocated_volume += can_allocate
                            if allocated_reduction_volume_year >= target_reduction_volume_year:
                                break
                elif constraint_type == "Budget Constrained":
                    for _, project in available_reductions.iterrows():
                        affordable_volume = (target_budget_reduction_year - allocated_cost_reduction_year) / (project.get(price_col, 0) + 1e-9)
                        can_allocate = min(affordable_volume, project.get(volume_col, 0))
                        cost = can_allocate * project.get(price_col, 0)
                        if cost > 0:
                            allocated_projects[project['project name']] = {'volume': can_allocate, 'price': project.get(price_col, 0), 'type': 'reduction'}
                            allocated_cost_reduction_year += cost
                            total_allocated_volume += can_allocate
                            total_allocated_cost += cost
                            if allocated_cost_reduction_year >= target_budget_reduction_year:
                                break

            portfolio[year] = allocated_projects

        # --- Portfolio Analysis and Visualization ---
        composition_by_type = {t: [] for t in all_types}
        avg_prices = []
        yearly_costs = {}
        yearly_volumes = {}

        for year in selected_years:
            total_cost = 0
            total_volume = 0
            type_volumes = {t: 0 for t in all_types}
            for project_info in portfolio[year].values():
                type_volumes[project_info['type']] += project_info['volume']
                total_cost += project_info['volume'] * project_info['price']
                total_volume += project_info['volume']

            for t in all_types:
                composition_by_type[t].append(type_volumes[t])

            avg_prices.append(total_cost / (total_volume + 1e-9) if total_volume > 0 else 0)
            yearly_costs[year] = total_cost
            yearly_volumes[year] = total_volume

        st.subheader("Portfolio Composition & Price Over Time")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for type_name in all_types:
            color = '#8BC34A' if type_name == 'technical removal' else '#AED581' if type_name == 'natural removal' else '#C5E1A5'
            fig.add_trace(go.Bar(x=selected_years, y=composition_by_type[type_name], name=type_name.capitalize(), marker_color=color), secondary_y=False)
        fig.add_trace(go.Scatter(x=selected_years, y=avg_prices, name='Average Price (€/tonne)', marker=dict(symbol='circle'), line=dict(color='#558B2F')), secondary_y=True)
        fig.update_layout(xaxis_title='Year', yaxis_title='Volume (tonnes)', yaxis2_title='Average Price (€/tonne)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='stack', template="plotly_white")
        st.plotly_chart(fig)

        final_tech = composition_by_type.get('technical removal', [0] * years)[-1]
        final_nat = composition_by_type.get('natural removal', [0] * years)[-1]
        final_total_removal = final_tech + final_nat
        final_reduction = composition_by_type.get('reduction', [0] * years)[-1]
        final_total = final_total_removal + final_reduction
        achieved_removal = (final_total_removal) / (final_total + 1e-9) * 100 if final_total > 0 else 0
        st.markdown(f"**Achieved Removal % in {end_year}: {achieved_removal:.2f}%**" if ('technical removal' in all_types or 'natural removal' in all_types) else "**No removal projects selected.**")

        st.subheader("Yearly Summary")
        summary_data = {'Year': selected_years}
        if constraint_type == "Volume Constrained":
            summary_data['Target Volume (tonnes)'] = [annual_constraints[year] for year in selected_years]
            summary_data['Achieved Volume (tonnes)'] = [yearly_volumes[year] for year in selected_years]
            summary_data['Total Cost (€)'] = [yearly_costs[year] for year in selected_years]
        else:
            summary_data['Budget (€)'] = [annual_constraints[year] for year in selected_years]
            summary_data['Total Cost (€)'] = [yearly_costs[year] for year in selected_years]
            summary_data['Achieved Volume (tonnes)'] = [yearly_volumes[year] for year in selected_years]
        st.dataframe(pd.DataFrame(summary_data))

        if broken_rules:
            st.warning("One or more constraints could not be fully satisfied:")
            for msg in broken_rules:
                st.text(f"- {msg}")

        if st.checkbox("Show raw project allocations"):
            full_table = []
            for year, projects in portfolio.items():
                for name, info in projects.items():
                    full_table.append({
                        'year': year,
                        'project name': name,
                        'type': info['type'],
                        'volume': info['volume'],
                        'price': info['price'],
                        'cost': info['volume'] * info['price']
                    })
            st.dataframe(pd.DataFrame(full_table))
