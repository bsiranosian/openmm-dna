# useful plotting functions for supercontact
import socket
import sys 

hn = socket.gethostname()
if hn == 'aspire':
    supercontactSrc = '/home/ben/projects/supercontact/src/'
elif hn == 'BoettigerServer':
    supercontactSrc = 'C:\\Users\\Ben\\projects\\supercontact\\src\\'
elif hn.startswith('sherlock') or os.getenv('SHERLOCK') == '1' or os.getenv('SHERLOCK') == '2':
    supercontactSrc = '~/projects/supercontact/src/'
else: 
    print('Warning: set location of supercontact repository src file in this script')
    supercontactSrc = '~/projects/supercontact/src/'

sys.path.append(supercontactSrc) 
sys.path.append(supercontactSrc + 'directional_model') 
from useful_3D import *


# from a simulation run trajectory, load up all 3D structures
# and plot an average distance and contact map for them
# options control which structures to load 
def plot_avDist_contact(inFolder, outFolder, r=10,
 startAt=1, loadN=1, saveFig=True,
  contactVmaxPercentile=97):
    manyStruc = load_run_trajectory(inFolder, startAt=startAt, loadN=loadN)
    manyDist = xyz_to_distance_many(manyStruc)
    avDist = np.mean(manyDist, axis=0)
    avContact = average_contact_from_distance_k562(manyDist, contact_threshold=r)

    # save average distance map
    plt.matshow(avDist, cmap='magma_r')
    plt.colorbar()
    plt.title('AvDist skip=' + str(loadN) + ' n=' + str(manyStruc.shape[0]), y=1.08)
    if saveFig:
        plt.savefig(join(outFolder, 'avDist.png'), bbox_inches='tight')
    else: 
        plt.show(block=False)
    # save average contact map
    plt.matshow(avContact, cmap='magma', vmax=np.nanpercentile(avContact, contactVmaxPercentile))
    plt.colorbar()
    plt.title('avContact skip=' + str(loadN) + ' n=' + str(manyStruc.shape[0]) +
     ' r='+ str(r)+'nm', y=1.08)
    if saveFig:
        plt.savefig(join(outFolder, 'avContact.png'), bbox_inches='tight')
    else: 
        plt.show(block=False)
    plt.close('all')