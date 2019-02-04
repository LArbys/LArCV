from __future__ import print_function
import os,sys

from larcv import larcv

if __name__ == "__main__":
    a = larcv.ImageMeta( 3456, 1008*6,
                         1008, 3456,
                         0, 2400, 0)

    b = larcv.ImageMeta( 100, 200,
                         100, 100,
                         50,  150, 0 )
    
    c = larcv.ImageMeta( 100, 200,
                         100, 100,
                         74, 170, 0 )

    inclusive = b.inclusive( c )

    print("meta a: {}".format(a.dump()))
    print("meta b: {}".format(b.dump()))
    print("meta c: {}".format(c.dump()))
    print()
    print("inclusive meta: {}".format(inclusive.dump()))
    
    if ( inclusive.min_x()!=50
         or inclusive.max_x()!=174
         or inclusive.min_y()!=150
         or inclusive.max_y()!=370
         or inclusive.pixel_width()!=1
         or inclusive.pixel_height()!=2
         or inclusive.rows()!=(370-150)/2
         or inclusive.cols()!=(174-50)/1 ):
        print("error: inclusive is not the expected size")
        sys.exit(-1)


    print("[metamethods] OK")
    sys.exit(0)
