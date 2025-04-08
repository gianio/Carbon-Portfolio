import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For potentially more appealing styles

# Set Streamlit page config with light green background via HTML/CSS
st.markdown("""
    <style>
    body {
        background-color: #e6f2e6;
    }
    .stApp {
        background-color: #e6f2e6;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Multi-Year Carbon Portfolio Builder")

# Upload CSV
df = st.file_uploader("Upload project CSV", type="csv")

if df:
    data = pd.read_csv(df)

    st.subheader("Project Overview")
    years = st.slider("How many years should the portfolio span?", 1, 6, 6)
    end_year = 2025 + years - 1
    selected_years = list(range(2025, end_year + 1))

    overview = data.copy()
    price_cols = [f"price {year}" for year in selected_years if f"price {year}" in data.columns]
    overview['avg_price'] = overview[price_cols].mean(axis=1)
    if 'description' in overview.columns:
        st.dataframe(overview[['project name', 'project type', 'description', 'avg_price']].drop_duplicates())
    else:
        st.dataframe(overview[['project name', 'project type', 'avg_price']].drop_duplicates())

    st.subheader("Step 1: Define Portfolio Settings")
    annual_volumes = {}
    for year in selected_years:
        annual_volumes[year] = st.number_input(f"Annual purchase volume for {year}:", min_value=0, step=100, value=1000)

    removal_target = st.slider(f"Target total removal % for year {end_year}", 0, 100, 80) / 100
    transition_speed = st.slider("Transition Speed (1 = Slow, 10 = Fast)", 1, 10, 5)
    removal_preference = st.slider("Removal Preference (1 = Natural, 10 = Technical)", 1, 10, 5)

    st.subheader("Step 2: Select Projects")
    project_names = data['project name'].unique().tolist()
    selected_projects = st.multiselect("Select projects to include in the portfolio:", project_names)

    if selected_projects:
        selected_df = data[data['project name'].isin(selected_projects)].copy()

        types = ['technical removal', 'natural removal', 'reduction']
        project_types = {t: selected_df[selected_df['project type'] == t] for t in types}

        removal_percentages = []
        for i, year in enumerate(selected_years):
            progress = (i + 1) / years
            factor = 0.5 + 0.1 * transition_speed
            removal_pct = removal_target * (progress ** factor)
            removal_percentages.append(min(removal_pct, removal_target))

        category_split = {
            'technical removal': removal_preference / 10,
            'natural removal': 1 - (removal_preference / 10)
        }

        portfolio = {year: {} for year in selected_years}
        broken_rules = []

        for year_idx, year in enumerate(selected_years):
            year_str = f"{year}"
            removal_share = removal_percentages[year_idx]
            reduction_share = 1 - removal_share
            annual_volume = annual_volumes[year]

            volumes = {}
            total_allocated = 0

            for category in types:
                if category == 'reduction':
                    category_share = reduction_share
                else:
                    category_share = removal_share * category_split[category]

                category_projects = project_types[category].copy()
                category_projects = category_projects[category_projects.get(f"available volume {year_str}", 0) > 0]

                if category_projects.empty:
                    broken_rules.append(f"No available volume for {category} in {year}.")
                    continue

                share_per_project = category_share / len(category_projects)
                for _, row in category_projects.iterrows():
                    max_available = row.get(f"available volume {year_str}", 0)
                    target_volume = annual_volume * share_per_project
                    vol = min(target_volume, max_available)
                    total_allocated += vol
                    key = row['project name']
                    volumes[key] = {
                        'volume': int(vol),  # Ensure volume is an integer
                        'price': row.get(f'price {year_str}', 0),
                        'type': category
                    }

            if total_allocated > 0:
                scale_factor = annual_volume / total_allocated
                for v in volumes.values():
                    v['volume'] = int(v['volume'] * scale_factor) # Scale and ensure integer
            else:
                broken_rules.append(f"No available projects in {year}, cannot allocate volume.")

            portfolio[year] = volumes

        composition_by_type = {t: [] for t in types}
        avg_prices = []

        for year in selected_years:
            annual_volume = annual_volumes[year]
            totals = {t: 0 for t in types}
            total_cost = 0
            for data in portfolio[year].values():
                totals[data['type']] += data['volume']
                total_cost += data['volume'] * data['price']

            for t in types:
                composition_by_type[t].append(totals[t])

            avg_prices.append(total_cost / annual_volume if annual_volume > 0 else 0)

        st.subheader("Portfolio Composition & Price Over Time")
        fig, ax1 = plt.subplots(figsize=(10, 6))
        sns.set_theme(style="whitegrid")  # Apply a clean whitegrid style
        green_palette = sns.color_palette("light:#558B2F", n_colors=3) # Define a green palette

        bar_width = 0.6
        x = np.arange(len(selected_years))

        tech = composition_by_type['technical removal']
        nat = composition_by_type['natural removal']
        red = composition_by_type['reduction']

        ax1.bar(x, tech, bar_width, label='Technical Removal', color=green_palette[0])
        ax1.bar(x, nat, bar_width, bottom=tech, label='Natural Removal', color=green_palette[1])
        ax1.bar(x, red, bar_width, bottom=np.array(tech) + np.array(nat), label='Reduction', color=green_palette[2])

        ax1.set_xlabel('Year')
        ax1.set_ylabel('Volume')
        ax1.set_xticks(x)
        ax1.set_xticklabels(selected_years)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        ax2.plot(x, avg_prices, marker='o', color=green_palette[0], label='Avg Price') # Use a green color for the line
        ax2.set_ylabel('Average Price')
        ax2.legend(loc='upper right')

        st.pyplot(fig)

        final_tech = composition_by_type['technical removal'][-1]
        final_nat = composition_by_type['natural removal'][-1]
        final_total = final_tech + final_nat + composition_by_type['reduction'][-1]
        achieved_removal = (final_tech + final_nat) / final_total if final_total > 0 else 0

        st.markdown(f"**Achieved Removal % in {end_year}: {achieved_removal * 100:.2f}%**")

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

        if st.checkbox("Show Box Plots"):
            # Assuming you have some numerical columns you want to visualize
            numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
            # Filter out 'avg_price' if it's a numerical column to avoid redundancy
            numerical_cols = [col for col in numerical_cols if col not in ['avg_price']]
            selected_box_cols = st.multiselect("Select columns for box plots:", numerical_cols)

            if selected_box_cols:
                num_cols = len(selected_box_cols)
                cols_per_row = 3  # Adjust as needed
                num_rows = (num_cols + cols_per_row - 1) // cols_per_row
                fig, axes = plt.subplots(num_rows, min(num_cols, cols_per_row), figsize=(15, 5 * num_rows))
                axes = np.ravel(axes)  # Flatten axes array for easy indexing

                for i, col in enumerate(selected_box_cols):
                    bp = axes[i].boxplot(data[col], patch_artist=True)
                    axes[i].set_title(col)

                    # Customize the boxes for rounded edges
                    for box in bp['boxes']:
                        box.set_linewidth(1)
                        box.set_edgecolor('black')
                        box.set_facecolor('lightgreen')
                        box.set_alpha(0.7)

                        # Get the path and modify it for rounded edges
                        path = box.get_path()
                        verts = path.vertices
                        codes = path.codes

                        # Add rounded corners using PathPatch with round style
                        import matplotlib.patches as patches
                        rounded_rect = patches.PathPatch(
                            verts,
                            codes=codes,
                            facecolor=box.get_facecolor(),
                            edgecolor=box.get_edgecolor(),
                            linewidth=box.get_linewidth(),
                            alpha=box.get_alpha(),
                            mutation_scale=1.0,
                            capstyle='round',
                            joinstyle='round'
                        )
                        axes[i].add_patch(rounded_rect)
                        box.set_visible(False) # Hide the original box

                    # Customize other elements (whiskers, caps, medians) if desired
                    for whisker in bp['whiskers']:
                        whisker.set(color='gray', linewidth=1)
                    for cap in bp['caps']:
                        cap.set(color='black', linewidth=1)
                    for median in bp['medians']:
                        median.set(color='red', linewidth=2)

                # Hide any unused subplots
                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                st.pyplot(fig)
