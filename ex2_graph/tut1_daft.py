from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import daft

# Instantiate the PGM, a directed graph
pgm = daft.PGM([8, 8], directed=True, node_unit=2,
               label_params=dict(fontsize=25))


pgm.add_node(daft.Node("x2", r"$x_2$", 1, 2))
pgm.add_node(daft.Node("x1", r"$x_1$", 4, 2))
pgm.add_node(daft.Node("x3", r"$x_3$", 7, 2))

pgm.add_node(daft.Node("x4", r"$x_4$", 2.5, 4))
pgm.add_node(daft.Node("x5", r"$x_5$", 5.5, 4))

pgm.add_node(daft.Node("x6", r"$x_6$", 2.5, 6))
pgm.add_node(daft.Node("x7", r"$x_7$", 5.5, 6))

# ====== start adding the edges ====== #

# p(x4 | x1, x2, x3)
pgm.add_edge("x1", "x4")
pgm.add_edge("x2", "x4")
pgm.add_edge("x3", "x4")

# p(x5 | x1, x3)
pgm.add_edge("x1", "x5")
pgm.add_edge("x3", "x5")

# p(x6 | x4)
pgm.add_edge("x4", "x6")

# p(x7 | x4, x5)
pgm.add_edge("x4", "x7")
pgm.add_edge("x5", "x7")


# this will save the figure to path: /tmp/tmp.png
pgm.figure.savefig("/tmp/tmp.png", dpi=250)
