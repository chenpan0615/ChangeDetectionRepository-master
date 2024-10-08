from Methodology.Traditional.MAD import run_irmad
from Methodology.Traditional.CVA import run_cva
from Methodology.Traditional.PCAKmeans import run_pcakmean
from Methodology.Traditional.ID import run_id
from Methodology.Traditional.Post_Classification import run_pc
from Methodology.Traditional.VID  import run_vid


if __name__ == '__main__':
    
    pre_data = './data/A/test.tif'
    post_data = './data/B/test.tif'
    pre_data_4b = './data/A/test_4b.tif'
    post_data_4b = './data/B/test_4b.tif'
    pre_lc = './data/A/lc.tif'
    post_lc = './data/B/lc.tif'
    
    run_cva(pre_data, post_data, './out/cva.png')
    
    run_irmad(pre_data, post_data, './out/irmad.png')
    run_id(pre_data, post_data, './out/id.png')
    run_vid(pre_data_4b, post_data_4b, './out/vid.png')
    run_pcakmean(pre_data, post_data, './out/pcakmean.png')
    run_pc(pre_lc, post_lc, './out/')