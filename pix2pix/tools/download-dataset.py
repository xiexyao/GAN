from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from urllib.request import urlretrieve
    from urllib.request import urlopen # python 3
except ImportError:
    from urllib2 import urlopen # python 2
import sys
import tarfile
import tempfile
import shutil
import os

dataset = sys.argv[1]
url = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/%s.tar.gz" % dataset
# with tempfile.TemporaryFile() as tmp:
#     print("downloading", url)
#     shutil.copyfileobj(urlopen(url), tmp)
#     print("extracting")
#     tmp.seek(0)
#     tar = tarfile.open(fileobj=tmp)
#     tar.extractall()
#     tar.close()
#     print("done")


######### 带进度条的下载
dest_file = os.path.join(sys.argv[2], "%s.gz" % sys.argv[1])
if not os.path.exists(dest_file):
    print("downloading", url)
    def _progress(count, block_size, total_size):
        sys.stdout.write("\rDownloading %0.1f%%" % ((count * block_size) / total_size * 100.0))
        sys.stdout.flush()

    filepath,_ = urlretrieve(url, dest_file, _progress)
    print()
print("extracting")
statinfo = os.stat(dest_file)
print('Successfully downloaded', url, statinfo.st_size, 'bytes.')
dest_directory = os.path.join(sys.argv[2])
if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
tarfile.open(dest_file, 'r:gz').extractall(dest_directory)
print("done")