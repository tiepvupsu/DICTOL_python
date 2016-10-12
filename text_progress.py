import time
import sys
import math
max_iter = 100
total_point = 50;
def progress_str(cur_val, max_val, total_point=50):
	p = int(math.ceil(float(cur_val)*total_point/ max_val))
	return '|' + p*'#'+ (total_point - p)*'.'+ '|' 
for i in range(100):
    time.sleep(.1)
    ## progress bar 
    str0 = progress_str(i, max_iter, total_point)
    sys.stdout.write("\r%s %d/%d" % (str0, i, max_iter ))
    sys.stdout.flush()
    print ''
