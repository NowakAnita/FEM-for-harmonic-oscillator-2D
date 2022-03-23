import class_for_md as md
import os 


def md_file(**kwargs):

    file_tit = kwargs["file_tit"]
    data_results = kwargs["data_results"]
    data_names = kwargs["data_names"]
    plots = kwargs["plots"]
    data_results_energy = kwargs["data_results_energy"]
    data_names_energy = kwargs["data_names_energy"]

    if os.path.exists(file_tit + '.md'):
        os.remove(file_tit + '.md')

    try:
        f = md.raport_md(open(file_tit + '.md', 'w'))

    except Exception as e:
        print(e)
        quit()

    # f.new_paragraph("Description:", test.text)
    # f.nl()

    f.header("Results:")
    f.table(data_names, data_results)

    f.header("Counted Energy:")
    f.table(data_names_energy, data_results_energy)

    if plots is not []:

        f.header("Task 6 - plots:")

        for name in plots:
            f.plot_im(name, '.png')
        
    
