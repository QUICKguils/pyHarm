from thesis import solveLazarusModel as sv

l = sv.main()
cont = l["cont_list"][-1]

plot_cont = sv.continuation_plotter()
plot_cont(cont, ih=[0], pred=True)
plot_cont(cont, ih=[1], pred=True)
plot_cont(cont, ih=[2], pred=True)
