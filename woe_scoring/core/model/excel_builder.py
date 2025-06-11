from typing import Dict, List, Any # Any for chart objects
import pandas as pd
import numpy as np # For np.cumsum

# Note: xlsxwriter specific objects like workbook, worksheet are not explicitly typed here
# but are used as per the library's API. 'Any' can be used for them if strict typing is needed.

def _format_main_scorecard_sheet(scorecard_worksheet: Any, # xlsxwriter worksheet object
                                 all_feature_stats_dfs: List[pd.DataFrame], # List of DFs from scorecard_builder
                                 cumulative_rows_per_feature: List[int], # Cumulative sum of rows for merge areas
                                 merge_cell_format: Any # xlsxwriter format object
                                 ) -> None:
    """Formats the main 'Scorecard' Excel sheet by merging cells for feature names, coefs, pvalues."""
    current_row_start_for_merge = 1 # xlsxwriter uses 0-indexed rows. If header is written by to_excel, data starts at row 1.
                                    # This variable will track the 0-indexed start row of the current data block.

    for i, feature_df in enumerate(all_feature_stats_dfs):
        if feature_df.empty:
            # Log: print(f"Warning: Skipping formatting for an empty feature DataFrame at index {i}.")
            continue

        # excel_data_end_row is the 0-indexed end row of the current block in the Excel sheet (data part)
        excel_data_end_row = cumulative_rows_per_feature[i] -1
        # So, indexes[i] is the end row for results[i] in the Excel sheet.
        # area_start was 1, then index+1. So, for first item, end is indexes[0]. For second, indexes[1].

        # The actual end row in Excel for this feature's block.
        # If full_features_df starts at Excel row 2 (due to header), then add 1.
        # Assuming full_features.to_excel writes header, so data starts at row 2.
        excel_data_end_row = cumulative_rows_per_feature[i] # This is the end row of the current feature block in the combined df
                                                            # relative to data start (0-indexed within data).
                                                            # For xlsxwriter, it's 1-indexed.

        # Values to merge are from the first row of the current feature's DataFrame
        try:
            # Assuming 'feature', 'coef', 'pvalue' are the first three columns after index, if written.
            # If to_excel(index=False), then these are df.columns[0], df.columns[1], df.columns[2]
            col_loc_feature = feature_df.columns.get_loc('feature')
            col_loc_coef = feature_df.columns.get_loc('coef')
            col_loc_pvalue = feature_df.columns.get_loc('pvalue')

            feature_name_to_merge = feature_df.iloc[0, col_loc_feature]
            coef_to_merge = feature_df.iloc[0, col_loc_coef]
            pvalue_to_merge = feature_df.iloc[0, col_loc_pvalue]
        except (KeyError, IndexError) as e:
            # Log: print(f"Warning: Could not get merge values for feature block {i} due to: {e}. Skipping this block's merge.")
            current_row_start_for_merge = excel_data_end_row + 1 # Move to next block
            continue

        values_to_merge_in_cols = [feature_name_to_merge, coef_to_merge, pvalue_to_merge]

        # Assuming to_excel(index=False) which means data columns A, B, C are 0, 1, 2
        for col_excel_0idx in range(3):
            value_to_fill_merged_cell = values_to_merge_in_cols[col_excel_0idx]

            # current_row_start_for_merge and excel_data_end_row are 0-indexed sheet rows for data
            if current_row_start_for_merge <= excel_data_end_row:
                try:
                    scorecard_worksheet.merge_range(
                        current_row_start_for_merge, col_excel_0idx,
                        excel_data_end_row,          col_excel_0idx,
                        value_to_fill_merged_cell,
                        merge_cell_format
                    )
                except Exception as e: # Catch potential xlsxwriter errors
                    # Log: print(f"Error merging cells for feature block {i}, col {col_excel_0idx}: {e}")
                    pass # Continue, try to format other parts

            widths = [20, 10, 10] # Corresponds to 'feature', 'coef', 'pvalue'
            try:
                scorecard_worksheet.set_column(col_excel_0idx, col_excel_0idx, widths[col_excel_0idx])
            except Exception as e:
                 # Log: print(f"Error setting column width for col {col_excel_0idx}: {e}")
                 pass

        current_row_start_for_merge = excel_data_end_row + 1 # Start of the next block


def _create_events_dist_chart(workbook: Any, chart_sheet_name: str, num_data_rows: int, col_name_to_idx_map: Dict[str, int]) -> Any:
    """Creates a combined column chart (event/non-event) and line chart (WOE)."""
    # Check for required columns for the chart
    required_cols = {'bin', 'event_cnt', 'non_event_cnt', 'WOE'}
    if not required_cols.issubset(col_name_to_idx_map.keys()):
        # Log: print(f"Warning: Missing one or more required columns {required_cols - set(col_name_to_idx_map.keys())} for event distribution chart on sheet '{chart_sheet_name}'. Skipping chart.")
        return None

    # Assuming to_excel(index=False) means 'bin' is col B (idx 1) if 'feature','coef','pval' are merged over it.
    # No, feature_df.to_excel(excel_writer, sheet_name=sheet_name, index=False) means columns of feature_df start at A.
    # So, col_name_to_idx_map gives the 0-indexed column letter.
    try:
        bin_col_letter = chr(ord('A') + col_name_to_idx_map['bin'])
        event_cnt_col_letter = chr(ord('A') + col_name_to_idx_map['event_cnt'])
        non_event_cnt_col_letter = chr(ord('A') + col_name_to_idx_map['non_event_cnt'])
        woe_col_letter = chr(ord('A') + col_name_to_idx_map['WOE'])
    except KeyError as e:
        # Log: print(f"Warning: Missing a required column key for event dist chart on sheet '{chart_sheet_name}': {e}. Skipping chart.")
        return None

    chart = workbook.add_chart({'type': 'column', 'subtype': 'stacked'})
    try:
        chart.add_series({
            'name':       f"='{chart_sheet_name}'!${event_cnt_col_letter}$1",
            'categories': f"='{chart_sheet_name}'!${bin_col_letter}$2:${bin_col_letter}${num_data_rows + 1}",
            'values':     f"='{chart_sheet_name}'!${event_cnt_col_letter}$2:${event_cnt_col_letter}${num_data_rows + 1}",
        })
        chart.add_series({
            'name':       f"='{chart_sheet_name}'!${non_event_cnt_col_letter}$1",
            'categories': f"='{chart_sheet_name}'!${bin_col_letter}$2:${bin_col_letter}${num_data_rows + 1}",
            'values':     f"='{chart_sheet_name}'!${non_event_cnt_col_letter}$2:${non_event_cnt_col_letter}${num_data_rows + 1}",
        })

        woe_line_chart = workbook.add_chart({'type': 'line'})
        woe_line_chart.add_series({
            'name':       f"='{chart_sheet_name}'!${woe_col_letter}$1",
            'categories': f"='{chart_sheet_name}'!${bin_col_letter}$2:${bin_col_letter}${num_data_rows + 1}",
            'values':     f"='{chart_sheet_name}'!${woe_col_letter}$2:${woe_col_letter}${num_data_rows + 1}",
            'y2_axis':    True,
            'smooth':     False,
        })
        chart.combine(woe_line_chart)
    except Exception as e: # Catch potential errors from add_series or combine
        # Log: print(f"Error creating event distribution chart for sheet '{chart_sheet_name}': {e}")
        return None # Return None if chart creation failed
    return chart

def _create_score_dist_chart(workbook: Any, chart_sheet_name: str, num_data_rows: int, col_name_to_idx_map: Dict[str, int]) -> Any:
    """Creates a combined column chart (score points) and line chart (event rate)."""
    required_cols = {'bin', 'score_ball', 'event_rate'}
    if not required_cols.issubset(col_name_to_idx_map.keys()):
        # Log: print(f"Warning: Missing one or more required columns {required_cols - set(col_name_to_idx_map.keys())} for score dist chart on sheet '{chart_sheet_name}'. Skipping.")
        return None

    try:
        bin_col_letter = chr(ord('A') + col_name_to_idx_map['bin'])
        score_ball_col_letter = chr(ord('A') + col_name_to_idx_map['score_ball'])
        event_rate_col_letter = chr(ord('A') + col_name_to_idx_map['event_rate'])
    except KeyError as e:
        # Log: print(f"Warning: Missing a required column key for score dist chart on sheet '{chart_sheet_name}': {e}. Skipping chart.")
        return None

    chart = workbook.add_chart({'type': 'column'})
    try:
        chart.add_series({
            'name':       f"='{chart_sheet_name}'!${score_ball_col_letter}$1",
            'categories': f"='{chart_sheet_name}'!${bin_col_letter}$2:${bin_col_letter}${num_data_rows + 1}",
            'values':     f"='{chart_sheet_name}'!${score_ball_col_letter}$2:${score_ball_col_letter}${num_data_rows + 1}",
        })

        event_rate_line_chart = workbook.add_chart({'type': 'line'})
        event_rate_line_chart.add_series({
            'name':       f"='{chart_sheet_name}'!${event_rate_col_letter}$1",
            'categories': f"='{chart_sheet_name}'!${bin_col_letter}$2:${bin_col_letter}${num_data_rows + 1}",
            'values':     f"='{chart_sheet_name}'!${event_rate_col_letter}$2:${event_rate_col_letter}${num_data_rows + 1}",
            'y2_axis':    True,
            'smooth':     False,
        })
        chart.combine(event_rate_line_chart)
    except Exception as e:
        # Log: print(f"Error creating score distribution chart for sheet '{chart_sheet_name}': {e}")
        return None # Return None if chart creation failed
    return chart

def _format_individual_feature_sheet(feature_sheet: Any, # xlsxwriter worksheet
                                     feature_data_df: pd.DataFrame,
                                     num_data_rows: int,
                                     merge_cell_format: Any) -> None:
    """Formats an individual feature's sheet, merging cells for feature name, coef, pvalue."""
    if feature_data_df.empty or num_data_rows == 0:
        # Log: print("Warning: Skipping formatting for empty individual feature sheet.")
        return

    try:
        # Assuming these columns exist as per scorecard_builder output
        col_loc_feature = feature_data_df.columns.get_loc('feature')
        col_loc_coef = feature_data_df.columns.get_loc('coef')
        col_loc_pvalue = feature_data_df.columns.get_loc('pvalue')

        feature_name_to_merge = feature_data_df.iloc[0, col_loc_feature]
        coef_to_merge = feature_data_df.iloc[0, col_loc_coef]
        pvalue_to_merge = feature_data_df.iloc[0, col_loc_pvalue]
    except (KeyError, IndexError) as e:
        # Log: print(f"Warning: Could not get merge values for individual feature sheet due to {e}. Skipping merge.")
        return

    values_to_merge_in_cols = [feature_name_to_merge, coef_to_merge, pvalue_to_merge]

    # Assuming to_excel(index=False) was used. Header is at xlsxwriter row 0. Data from row 1.
    excel_data_start_row_0idx = 1 # First data row (0-indexed for xlsxwriter)
    excel_data_end_row_0idx = num_data_rows # Last data row (0-indexed for xlsxwriter)

    for col_data_idx in range(3): # For 'feature', 'coef', 'pvalue'
        col_excel_0idx = col_data_idx # These are the first three columns A, B, C
        value_to_fill = values_to_merge_in_cols[col_data_idx]

        # Merge from the first data row to the last data row
        if excel_data_start_row_0idx <= excel_data_end_row_0idx :
            try:
                feature_sheet.merge_range(
                    excel_data_start_row_0idx, col_excel_0idx,
                    excel_data_end_row_0idx,   col_excel_0idx,
                    value_to_fill,
                    merge_cell_format
                )
            except Exception as e:
                # Log: print(f"Error merging cells for individual sheet, col {col_excel_0idx}: {e}")
                pass

        widths = [20, 10, 10]
        feature_sheet.set_column(col_excel_0idx, col_excel_0idx, widths[col_data_idx])


def _add_feature_charts_to_excel(feature_stats_dfs: List[pd.DataFrame], # List of DFs, one per feature
                                 excel_writer: pd.ExcelWriter,
                                 workbook: Any, # xlsxwriter workbook
                                 chart_width: int, chart_height: int,
                                 first_chart_pos_excel: str, second_chart_pos_excel: str) -> None:
    """Adds data and charts for each feature to its own sheet in the Excel file."""
    merge_format = workbook.add_format({'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter'})

    for feature_df in feature_stats_dfs:
        if feature_df.empty or feature_df.iloc[0]['feature'] == 'const':
            continue # Skip 'const' row or empty DFs for individual sheets/charts

        # Use original feature name for sheet name (already cleaned in scorecard_builder)
        sheet_name = str(feature_df.iloc[0]['feature'])
        # Ensure sheet name is valid for Excel (e.g., <= 31 chars, no invalid chars)
        clean_sheet_name = str(feature_df.iloc[0]['feature'])
        clean_sheet_name = clean_sheet_name.replace('[','').replace(']','').replace('*','').replace(':','').replace('?','').replace('/','\\')[:31]

        try:
            feature_df.to_excel(excel_writer, sheet_name=clean_sheet_name, index=False)
            feature_sheet = excel_writer.sheets[clean_sheet_name]
        except Exception as e:
            # Log: print(f"Error writing feature_df to sheet '{clean_sheet_name}': {e}. Skipping this sheet.")
            continue # Skip to next feature if sheet creation fails

        num_data_rows = len(feature_df)

        col_name_to_idx_map = {col_name: i for i, col_name in enumerate(feature_df.columns)}

        events_chart = _create_events_dist_chart(workbook, clean_sheet_name, num_data_rows, col_name_to_idx_map)
        score_chart = _create_score_dist_chart(workbook, clean_sheet_name, num_data_rows, col_name_to_idx_map)

        try:
            if events_chart:
                events_chart.set_size({'width': chart_width, 'height': chart_height})
                events_chart.set_legend({'position': 'bottom'})
                feature_sheet.insert_chart(f'{first_chart_pos_excel}{num_data_rows + 3}', events_chart)

            if score_chart:
                score_chart.set_size({'width': chart_width, 'height': chart_height})
                score_chart.set_legend({'position': 'bottom'})
                feature_sheet.insert_chart(f'{second_chart_pos_excel}{num_data_rows + 3}', score_chart)
        except Exception as e:
            # Log: print(f"Error setting chart properties or inserting chart for sheet '{clean_sheet_name}': {e}")
            # Continue even if charts fail for this sheet
            pass

        _format_individual_feature_sheet(feature_sheet, feature_df, num_data_rows, merge_format)


def build_excel_scorecard_sheet(
    all_feature_scorecard_data: List[pd.DataFrame], # List of DFs from scorecard_builder
    excel_writer: pd.ExcelWriter,
    chart_width: int = 640,
    chart_height: int = 480,
    first_plot_col_letter: str = 'A', # Column letter for first chart on individual sheets
    second_plot_col_letter: str = 'J' # Column letter for second chart
    ) -> None:
    """
    Builds the complete Excel scorecard, including a main sheet with all features
    and individual sheets for each feature with charts.
    """
    workbook = excel_writer.book
    main_sheet_merge_format = workbook.add_format({
        'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'text_wrap': True
    })

    # Separate 'const' row from actual feature data for specific handling if needed
    const_df_list = [df for df in all_feature_scorecard_data if not df.empty and df.iloc[0]['feature'] == 'const']
    actual_feature_dfs = [df for df in all_feature_scorecard_data if not df.empty and df.iloc[0]['feature'] != 'const']

    # Concatenate all DataFrames (const first, then features) for the main 'Scorecard' sheet
    # Ensure 'const' df is handled correctly if its structure differs slightly (e.g. fewer columns for bin details)
    # For pd.concat, all DFs should ideally have same columns. Scorecard_builder ensures this by padding with '-'.
    if not all_feature_scorecard_data: # Handle empty input
        # Create an empty sheet or write a message
        empty_df = pd.DataFrame({'Message': ['No scorecard data to write.']})
        empty_df.to_excel(excel_writer, sheet_name='Scorecard', index=False)
        return

    full_features_summary_df = pd.concat(all_feature_scorecard_data, ignore_index=True)
    full_features_summary_df.to_excel(excel_writer, sheet_name='Scorecard', index=False)

    scorecard_main_sheet = excel_writer.sheets['Scorecard']

    # Calculate cumulative row counts for merging cells in the main sheet
    # This needs to be based on the DFs as they were before concat for accurate row counts per feature block
    # Lengths are number of bins + 1 header for each feature, or just 1 for const
    # The DataFrame written (full_features_summary_df) has its own index.
    # `to_excel` writes a header row. So, data starts at Excel row 2.
    # `cumulative_rows_per_feature` for _format_main_scorecard_sheet.
    # This should be 0-indexed end row for each block within the data part of the sheet.
    # If to_excel(index=False) writes header on row 0 (xlsxwriter 0-indexed), data starts on row 1 (xlsxwriter 0-indexed).

    current_data_row_count = 0 # Tracks number of data rows written so far (0-indexed)
    cumulative_data_end_rows_0idx = []
    for df in all_feature_scorecard_data:
        # Add 1 for header row written by to_excel for each df if they were written separately.
        # But here, we concat then write once. So just len(df).
        current_data_row_count += len(df)
        # This is the 0-indexed end row of the data block in the sheet (assuming data starts at row 1)
        cumulative_data_end_rows_0idx.append(current_data_row_count)

    try:
        _format_main_scorecard_sheet(
            scorecard_main_sheet,
            all_feature_scorecard_data,
            cumulative_data_end_rows_0idx, # Pass 0-indexed end rows relative to data start (row 1)
            main_sheet_merge_format
        )
    except Exception as e:
        # Log: print(f"Error formatting main scorecard sheet: {e}. Proceeding without full formatting.")
        pass # Allow to proceed to individual sheets

    # Add individual feature sheets with charts (only for actual features, not 'const')
    # This part already contains try-except for individual sheet/chart errors.
    _add_feature_charts_to_excel(actual_feature_dfs, excel_writer, workbook,
                                 chart_width, chart_height,
                                 first_plot_col_letter, second_plot_col_letter)
