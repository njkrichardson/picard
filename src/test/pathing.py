from os.path import abspath, dirname
import sys 

src_ = dirname(dirname(abspath(__file__)))
sys.path.append(src_) 
