import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

outf = 'test.avi'
rate = 1

cmdstring = ('local/bin/ffmpeg',
             '-r', '%d' % rate,
             '-f','image2pipe',
             '-vcodec', 'png',
             '-i', 'pipe:', outf
             )
p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

plt.figure()
frames = 10
for i in range(frames):
    plt.imshow(np.random.randn(100,100))
    plt.savefig(p.stdin, format='png')