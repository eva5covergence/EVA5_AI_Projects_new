import hiddenlayer as hl

def draw_graph(model, input, view="TD"):
  graph = hl.build_graph(model, input)
  dot=graph.build_dot()
  dot.attr("graph", rankdir="TD") if view=="TD" else dot.attr("graph", rankdir="LR")
  return dot
