from optparse import OptionParser

from PIL import Image
import numpy as np

from seam.retarget import retargetRowfirst, retargetOptimal, retargetColfirst


def main():
    usage = "usage: %prog [options] input_name output_column output_row energy_type output_name"
    parser = OptionParser(usage)
    parser.add_option('-o', '--order', help='the seam order', action='store', type='string', default='col', dest='order')
    parser.add_option('-r', '--ratio', help='the ratio of gradient', action='store', type='float', default=0.5, dest='gdratio')
    parser.add_option('-n', '--net', help='the deep network used', action='store', type='string', default='squeezenet', dest='net')
    options, args = parser.parse_args()
    print(options)
    if len(args) != 5:
        parser.error('incorrect number of arguments')
    filename = args[0]
    outputname = args[4]
    try:
        columns = int(args[1])
        rows = int(args[2])
        energyType = int(args[3])

        img = Image.open(filename)
        npimg = np.array(img)

        print('For better performance on large images, we didn\'t compile the numba modules ahead of time, so please wait when compiling the modules')
        if options.order == 'opt':
            r, c, _ = npimg.shape
            if r >= rows and columns >= c:
                npout = retargetOptimal(npimg, options.gdratio, rows, columns)
            else:
                print('Optimal order only for removal, use column first instead')
                npout = retargetColfirst(npimg, options.gdratio, rows, columns)
        elif options.order == 'row':
            npout = retargetRowfirst(npimg, options.gdratio, rows, columns)
        elif options.order == 'col':
            npout = retargetRowfirst(npimg, options.gdratio, rows, columns)
        else:
            parser.error('order should be col, row or opt')
        newimg = Image.fromarray(npout)
        newimg.save(outputname)
    except ValueError as e:
        parser.error('should receive integer arguments')
    except FileNotFoundError as e:
        parser.error('input file not exist')




if __name__ == '__main__':
    main()