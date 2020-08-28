from bokeh.plotting import ColumnDataSource, output_file, save
from bokeh.models import CustomJS, TextInput, DataTable, TableColumn
from bokeh.layouts import column, row

def create_nn_finder_html(distances, text_df,  file_path):

    source = ColumnDataSource(data=dict(
        ids=range(len(text_df.text)),
        distances=distances.tolist(),
        text=text_df.text,
        display_text=text_df.text,
        display_ids=range(len(text_df.text)),
    ))

    display_source = ColumnDataSource(data=dict(
        closest_text=[""] * 20,
        closest_dist=[0] * 20,
    ))

    columns = [
        TableColumn(field="display_text", title="Text"),
    ]

    closest_columns = [
        TableColumn(field="closest_text", title="Closest examples"),
        TableColumn(field="closest_dist", title="Distance", width=20),
    ]

    str_search_input = TextInput(value="", title="Search feedback")

    callback = CustomJS(args=dict(source=source, display_source=display_source, search_text=str_search_input),
                    code="""
        const data = source.data;

        // ##################
        // First search
        // ##################

        const search_text_str = search_text.value.toLowerCase();

        const display_texts = [];
        const display_ids = [];

        data['text'].map(function(e, i) {
            const text_val = data['text'][i];
            const text_id = data['ids'][i];
            if (text_val.toLowerCase().includes(search_text_str)){
                display_texts.push(text_val);
                display_ids.push(text_id);
            }
        });

        data['display_text'] = display_texts;
        data['display_ids'] = display_ids;

        source.change.emit();

        // ##################
        // Then show selected
        // ##################

        if(source.selected.indices.length >= 1){

            const selected_table_idx = source.selected.indices[0];

            if (selected_table_idx >= data['display_ids'].length){
                console.log("Empty cell selected")
            }else{
                const selected_idx = data['display_ids'][selected_table_idx];

                console.log(selected_idx)

                const texts = data['text'];

                const list_of_dist = data['distances'];

                const selected_dist = list_of_dist[selected_idx];

                function indexOfNMin(arr, n) {

                    if (arr.length < n) {
                        return [-1];
                    }

                    var min_arr = arr.slice(0, n);
                    var min_idxs = [...Array(n).keys()];

                    for (var i = n; i < arr.length; i++) {
                        max_selected = Math.max(...min_arr);

                        if (arr[i] < max_selected) {
                            var idx_max = min_arr.indexOf(max_selected);
                            min_arr[idx_max] = arr[i];
                            min_idxs[idx_max] = i;
                        }
                    }

                    return [min_arr, min_idxs];
                }

                const closest_dist_values = indexOfNMin(selected_dist, 20);
                const closest_dist =  [].slice.call(closest_dist_values[0]);
                const closest_dist_idx = closest_dist_values[1];

                function sortWithIndices(inputArray) {

                    const toSort = inputArray.slice();

                    for (var i = 0; i < toSort.length; i++) {
                        toSort[i] = [toSort[i], i];
                    }

                    toSort.sort(function(left, right) {
                        return left[0] < right[0] ? -1 : 1;
                    });

                    var sortIndices = [];

                    for (var j = 0; j < toSort.length; j++) {
                        sortIndices.push(toSort[j][1]);
                    }

                    return sortIndices;
                }

                const sorted_closest_dist_idx_idx = sortWithIndices(closest_dist);

                const sorted_closest_dist_idx = sorted_closest_dist_idx_idx.map(i => closest_dist_idx[i]);

                const closest_texts = sorted_closest_dist_idx.map(i => texts[i]);

                const display_data = display_source.data;
                display_data['closest_text'] = closest_texts;
                display_data['closest_dist'] = closest_dist.sort(function(a, b){return a - b}).map(i => i.toFixed(3));

                display_source.change.emit();
            }
        }
    """)

    source.selected.js_on_change('indices', callback)
    str_search_input.js_on_change('value', callback)

    data_table = DataTable(source=source, columns=columns, width=600, height=420, selectable=True)
    closest_data_table = DataTable(source=display_source, columns=closest_columns, width=800, height=800, selectable=False)

    title = Div(text="""<b>Feedback Finder</b><br><br>
    The left hand side will allow you to look at ALL feedback for this given app.<br><br>
    Click on a row to see the closest matches to this row (and the embedding distance of each match) on the right side.<br><br>
    Try using the search bar to narrow down feedback that you want to find. <br>For example, if you are looking for performance related bug reports, then try typing 'lag' into the search bar, and hitting enter.<br> Then click on one of the results on the left to see other related bits of feedback that do not explicitly mention the word 'lag' on the right.<br><br>""",
    width=1000, height=180)

    layout = column(
        title,
        row(
            column(str_search_input, data_table),
            column(closest_data_table)
        )
    )

    # output to static HTML file
    output_file(f"{file_path}.html")

    save(layout)
