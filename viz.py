from .scenario_tree import *

from bokeh.plotting import figure
from bokeh.models import LabelSet, HoverTool,ColumnDataSource,CustomJSTransform
from bokeh.transform import transform
import bokeh.palettes as pal

def plot_tree(tree):
    line_len = 7
    width = 2
    sep = line_len + width
    mg = 0

    get_rect_tree_params(tree.root, 0, 0, 1, mg, sep)
    get_level(tree.root, 0)
    get_cond_prob(tree.root, 1)

    rects = pd.DataFrame.from_records(map(lambda x:
                                          {'x': x.cx, 'y': x.cy, 'width': width, 'height': x.l,
                                           'color': pal.Spectral11[x.lv], 'level': x.lv,
                                           'weight': round(x.weight, 2),
                                           'cond_Prob': x.cond_prob,
                                           'index': x.index,
                                           'value': x.value,
                                           'html': '<it>good</it>'
                                           }, tree.node_list))
    rects = ColumnDataSource(rects)

    lines_x = list(map(lambda x: [x.cx - line_len - width / 2, x.cx - width / 2], tree.node_list[1:]))
    lines_y = list(map(lambda x: [x.cy, x.cy], tree.node_list[1:]))
    x_tick_pos = list(set(rects.data['x']))
    x_tick = list(set(rects.data['level']))
    x_tick_pos.sort()
    x_tick.sort()

    p = figure(plot_width=600, plot_height=400, y_range=(-0.55, 0.55), tools="save")
    # title='Box Visualisation of A Scenario Tree')
    p.toolbar.logo = None
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.xaxis.axis_label = 'Time Step'
    p.yaxis.visible = False
    p.outline_line_color = None
    p.xaxis.ticker = x_tick_pos
    p.xaxis.major_label_overrides = {int(p): str(t)
                                     for p, t in zip(x_tick_pos, x_tick)}

    g_rect = p.rect(x='x', y='y', height='height',
                    width='width', alpha=0.3,
                    color='color', line_color='black',
                    line_width=1,
                    source=rects)

    trans = CustomJSTransform(
        v_func='''
            const norm = new Array(xs.length)
            for (let i = 0; i < xs.length; i++) {
                norm[i] = xs[i].toString()
            }
            return norm
        '''
    )

    labels = LabelSet(x='x', y='y', text=transform('value', trans), level='glyph',
                      x_offset=-10, y_offset=-4,
                      source=rects,
                      render_mode='canvas',
                      text_font_size="10pt")

    rect_hover = HoverTool(renderers=[g_rect],
                           tooltips=[("Time", "@level"),
                                     ("Prob", "@weight{1.11}",),
                                     ("cond.Prob", "@cond_Prob{1.11}")
                                     ]
                           )

    p.add_tools(rect_hover)
    p.multi_line(xs=lines_x, ys=lines_y,
                 line_width=0.5, color='black')
    p.add_layout(labels)

    return p