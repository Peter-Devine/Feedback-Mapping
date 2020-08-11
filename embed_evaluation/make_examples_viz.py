from bokeh.plotting import ColumnDataSource, figure, output_file, show
from bokeh.palettes import Turbo256
from bokeh.models import CustomJS, Slider, Select,TextInput
from bokeh.layouts import column, row

def create_example_viz_html(df, file_path):

    get_256_ratio = lambda x: int(256*x/len(df.label.unique()))
    colour_map = {lab: Turbo256[get_256_ratio(i)] for i, lab in enumerate(df.label.unique())}

    source = ColumnDataSource(data=dict(
        euc_x=df.euclidean_distance,
        cos_x=df.cosine_distance,
        display_x=df.euclidean_distance,
        text=df.text,
        label=df.label,
        color=[colour_map[x] for x in df.label]
    ))

    TOOLTIPS = [
        ("Distance", "@display_x"),
        ("Text", "@text"),
        ("Label", "@label"),
    ]

    total_num_points = df.shape[0]
    points_slider = Slider(start=1, end=total_num_points, value=30, step=1, title="Number of points")
    distance_select = Select(value='Euclidean', options=["Euclidean", "Cosine"])
    str_search_input = TextInput(value="", title="Search feedback")

    callback = CustomJS(args=dict(source=source, num_p=points_slider, distance_type=distance_select, search_text=str_search_input),
                        code="""
        const data = source.data;
        const n = num_p.value;
        const d_type = distance_type.value;
        const search_text_str = search_text.value.toLowerCase();

        if (d_type == "Euclidean") {
            data['display_x'] = data['euc_x'].slice(0, n);
        } else if (d_type == "Cosine") {
            data['display_x'] = data['cos_x'].slice(0, n);
        } else {
            throw 'Unsupported distance type: ' + d_type;
        }

        data['display_x'] = data['display_x'].map(function(e, i) {
            const text_val = data['text'][i];
            if (text_val.toLowerCase().includes(search_text_str)){
                return e
            }
        });

        source.change.emit();
    """)


    points_slider.js_on_change('value', callback)
    distance_select.js_on_change('value', callback)
    str_search_input.js_on_change('value', callback)

    p = figure(plot_width=900, plot_height=400, tooltips=TOOLTIPS,
               title=f"Points relative to '{df.text.iloc[0]}'")

    p.circle('display_x', size=18, source=source, fill_color="color")

    layout = row(
        p,
        column(points_slider, distance_select, str_search_input),
    )

    # output to static HTML file
    output_file(f"{file_path}.html")
