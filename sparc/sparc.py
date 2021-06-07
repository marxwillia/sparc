import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
import sys

sys.path.append('/Users/marcwilliamson/src/dev/SESNspectraPCA/code')
import SNIDsn
import SNIDdataset as snid

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import plotly.express as px
import plotly.graph_objs as go
import plotly.tools as tls



def SNIDdataset2DataFram(snidset):
    """
    Converts a SNIDdataset object into a pandas DataFrame.
    """
    snnames = []
    snphases = []
    specdata = []
    for snname in snidset.keys():
        #print(snname)
        snobj = snidset[snname]
        phases = snobj.phases
        phkeys = snobj.data.dtype.names
        #for ph in phases:
        for i, ph in enumerate(phkeys):
            #key = 'Ph'+str(ph)
            key = ph
            flux = snobj.data[key]
            snnames.append(snname)
            #snphases.append(ph)
            snphases.append(phases[i])
            specdata.append(flux)
            
    df = pd.DataFrame(data=np.array(specdata))
    df.insert(loc=0, column='SN', value=np.array(snnames))
    df.insert(loc=1, column='phase', value=np.array(snphases))
    return df
    
    
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 3, x.max() + 3
    y_min, y_max = y.min() - 3, y.max() + 3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy
    

    
class Sparc:
    
    def __init__(self, sparcpca, sparcsvm, obs_path):
        self.sparcpca = sparcpca
        self.sparcsvm = sparcsvm
        
        dat = np.loadtxt(obs_path)
        obs_wvl = dat[:,0]
        obs_spec = dat[:,1]
        
        
        return


    
    
class SparcSVM:
    
    def __init__(self, sparcpca, pcax, pcay, ncv):
        """
        Creates an SVM based classifier for a user
        specified SparcPCA object.
        """
        
        self.sparcpca = sparcpca
        self.ncv = ncv
        self.pcax = pcax
        self.pcay = pcay
        self.x = self.sparcpca.sn_pca_coeffs_np[:, pcax - 1]
        self.y = self.sparcpca.sn_pca_coeffs_np[:, pcay - 1]
        
        truth = np.zeros(self.sparcpca.rawdata.shape[0])
        for i, label in enumerate(self.sparcpca.labels):
            truth += i * (sparcpca.rawdata['sntype'] == label).values
        self.truth = truth
        
        return
    
    def train2dSVM(self):
        
        svm_arr = []
        
        dat = np.column_stack((self.x,self.y))
        
        for i in range(self.ncv):
            #linsvm = LinearSVC()
            linsvm = SVC(kernel='linear', max_iter=2000, probability=True)
            trainX, testX, trainY, testY = train_test_split(dat, self.truth, test_size=0.3)
            linsvm.fit(trainX, trainY)
            svm_arr.append(linsvm)
            
        self.svm_arr = svm_arr
        return
    
    

#traceIIb=go.Scatter(x=pcax[IIbmask], y=pcay[IIbmask], mode='markers',\
#                            marker=dict(size=10, line=dict(width=1), color=col_green, opacity=1), \
#                            text=np.array([nm+'_'+ph for nm,ph in zip(self.pcaNames, self.pcaPhases)])[IIbmask], name='IIb')    
    def plotlySVM(self):
        
        
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(layout=layout)
        Z_arr = []
        
        for svm in self.svm_arr:
            mesh_x, mesh_y = make_meshgrid(self.x, self.y, h=0.02)
            Z = svm.predict(np.c_[mesh_x.ravel(), mesh_y.ravel()])
            Z_arr.append(Z)
            labels_predicted = []
            for i, label in enumerate(self.sparcpca.labels):
                if len(Z[Z==i]) > 0:
                    labels_predicted.append(i)
            #colorscale = [[i/(len(labels_predicted)-1), self.sparcpca.colors[i]] for i in labels_predicted]
            colorscale = [[i/(len(self.sparcpca.labels) - 1), self.sparcpca.colors[i]] for i in range(len(self.sparcpca.labels))]
            print(colorscale)
            #contour_i = go.Contour(z=Z.reshape(mesh_x.shape),
            #                       x=mesh_x[0],
            #                       y=mesh_y[:,0],
            #                       colorscale=colorscale,
            #                       opacity=0.5/self.ncv,
            #                       showscale=False)
            #fig.add_trace(contour_i)
        Z_arr = np.array(Z_arr)
        Z_avg = np.median(Z_arr, axis=0)
        colorscale = [[i/(len(self.sparcpca.labels) - 1), self.sparcpca.colors[i]] for i in range(len(self.sparcpca.labels))]
        contour = go.Contour(z=Z_avg.reshape(mesh_x.shape),
                                   x=mesh_x[0],
                                   y=mesh_y[:,0],
                                   colorscale=colorscale,
                                   opacity=0.5,
                                   showscale=False)
        fig.add_trace(contour)
        
        
        
        pca_groupby = self.sparcpca.pca_df.groupby(by='sntype')
        for i, label in enumerate(self.sparcpca.labels):
            g = pca_groupby.get_group(label)
            col = self.sparcpca.colors[i]
            marker_text = [self.sparcpca.pca_df.iloc[ind]['SN'] + \
                           ', Age = '+str(self.sparcpca.pca_df.iloc[ind]['phase']) \
                           for ind in self.sparcpca.pca_df.index]
            trace_i = go.Scatter(mode='markers',
                                 x=g.loc[:, self.pcax - 1], y=g.loc[:, self.pcay - 1],
                                 marker=dict(color=col, size=10), 
                                 text=marker_text,
                                 name=label)
            fig.add_trace(trace_i)
            
            
    
        return fig
        
    
    
    def plotSVM(self):
        fig = plt.figure()
        ax = plt.gca()
        
        for svm in self.svm_arr:
            mesh_x, mesh_y = make_meshgrid(self.x, self.y, h=0.02)
            Z = svm.predict(np.c_[mesh_x.ravel(), mesh_y.ravel()])
            labels_predicted = []
            for i, label in enumerate(self.sparcpca.labels):
                if len(Z[Z==i]) > 0:
                    labels_predicted.append(i)
            nbins = len(labels_predicted)
            cmap_name = 'mymap'
            colors = [self.sparcpca.colors[ind] for ind in labels_predicted]
            cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
            Z = Z.reshape(mesh_x.shape)
            out = ax.contourf(mesh_x, mesh_y, Z, cmap=cm, alpha=0.2/self.ncv)
            
        pca_groupby = self.sparcpca.pca_df.groupby(by='sntype')
        for i, label in enumerate(self.sparcpca.labels):
            g = pca_groupby.get_group(label)
            col = self.sparcpca.colors[i]
            ax.scatter(g.loc[:, self.pcax - 1], g.loc[:, self.pcay - 1], c=col)
            
        
        return fig, ax

class SparcPCA:
    
    def __init__(self, datasets, labels, colors, wavelengths):
        """
        Loads SNIDdatasets provided by the user, along
        with user defined labels and colors for each 
        SNIDdataset.
        """
        rawdata = pd.DataFrame()
        for ds, lab in zip(datasets, labels):
            df = SNIDdataset2DataFram(ds)
            #types = np.array([lab]*len(ds.keys()))
            types = np.array([lab]*df.shape[0])
            df.insert(loc=1, column='sntype', value=types)
            rawdata = pd.concat([rawdata, df])
        
        rawdata.index = np.arange(rawdata.shape[0])
        self.rawdata = rawdata
        self.wavelengths = wavelengths
        self.labels = labels
        self.colors = colors
        
        return
    
    def plotlyMatch(self, obsdata):

        match_df = self.match(obsdata)
        
        layout = go.Layout(paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')
        fig = go.Figure(layout=layout)
        
        fig.add_trace(go.Scatter(x=self.wavelengths, y=obsdata,
                                 mode='lines', name='Observed Spectrum'))
        
        return fig
        
        
        
    
    def match(self, obsdata):
        """
        Calculate the matches of new data to 
        the trained templates. New data is 
        preprocessed consistently with the 
        training data and set to the same
        wavelength grid.
        """
        
        # TODO: Preprocess new data
        
        newPCAcoords = self._projectNewData(obsdata)
        distances = self._pcaDistance(newPCAcoords)
        match_df = pd.DataFrame()
        match_df.insert(loc=0, column='SN', value=self.rawdata['SN'])
        match_df.insert(loc=1, column='sntype', value=self.rawdata['sntype'])
        match_df.insert(loc=2, column='phase', value=self.rawdata['phase'])
        match_df.insert(loc=3, column='distance', value=distances)
        match_df = match_df.sort_values(by='distance', ascending=1)
        return match_df
    
    def _projectNewData(self, flux):
        """
        Calculates the projected PCA coefficients
        for new data using the existing eigenvectors
        (i.e. PCA is NOT rerun with the new data).
        This is a private helper function that ASSUMES
        the fluxes have been preprocessed to match
        the conditions of the training data and set to 
        the same wavelength grid.
        """
        
        return np.dot(self.evecs, flux.reshape((1,-1)).T).T
    
    def _pcaDistance(self, pcacoords):
        """
        Calculates the distance from given coordinates
        to all the training data points. Distance is
        weighted by cumulative variance explained by
        the corresponding eigenvector.
        """
        return np.dot(((self.sn_pca_coeffs_np - pcacoords)**2), (1-self.evals_cs))
        
        
        
    
    def runPCA(self):
        """
        Runs PCA and saves the eigenspectra, eigenvalues,
        and calculates the PCA coefficients for the SN data.
        """
        pca = PCA()
        pca.fit(self.raw_flux_np)
        self.evecs = pca.components_
        self.evals = pca.explained_variance_ratio_
        self.evals_cs = self.evals.cumsum()
        
        raw_sn_pca_coeffs = np.dot(self.evecs, self.raw_flux_np.T).T
        pca_df = pd.DataFrame(raw_sn_pca_coeffs)
        pca_df.insert(loc=0, column='SN', value=self.rawdata['SN'].values)
        pca_df.insert(loc=1, column='sntype', value=self.rawdata['sntype'].values)
        pca_df.insert(loc=2, column='phase', value=self.rawdata['phase'].values)
        self.pca_df = pca_df
        return
    
    def reconstructSN(self, df_row_index, nevecs):
        """
        Uses the PCA eigenspectra to reconstruct the original
        spectrum.
        """
        fig = plt.figure()
        datasetmean = np.mean(self.raw_flux_np, axis=0)
        truespec = self.raw_flux_np[df_row_index]
        pcacoeff = np.dot(self.evecs, (truespec - datasetmean))
        snname = self.rawdata.iloc[df_row_index]['SN']
        ph = self.rawdata.iloc[df_row_index]['phase']
        plt.plot(self.wavelengths, truespec, label=snname +' '+str(ph)+' days')
        plt.plot(self.wavelengths, datasetmean + (np.dot(pcacoeff[:nevecs], self.evecs[:nevecs])),
                 label='PCA reconstruction: %d evecs'%(nevecs))
        plt.xlabel("Wavelength ($\AA$)")
        plt.ylabel("Relative Flux")
        plt.legend()
        return fig
    
    def pcaScatter(self, pcax, pcay):
        
        
        fig = plt.figure(figsize=(10,7))
        ax = plt.gca()
        
        pca_groupby = self.pca_df.groupby(by='sntype')
        for i, label in enumerate(self.labels):
            g = pca_groupby.get_group(label)
            col = self.colors[i]
            ax.scatter(g.loc[:, pcax - 1], g.loc[:, pcay - 1], c=col, label=label)

        plt.xlabel('PCA%d'%(pcax), fontsize=20)
        plt.ylabel('PCA%d'%(pcay), fontsize=20)
        ax.legend(fontsize=20)
        return fig
    
    @property
    def raw_flux_np(self):
        """
        Returns only the fluxes from the SparcPCA
        DataFrame in a numpy 2d array.
        """
        return self.rawdata.iloc[:, 3:].values
    
    @property
    def sn_pca_coeffs_np(self):
        """
        Returns only the PCA coeffs for the SN
        data as a 2d numpy array.
        """
        return self.pca_df.iloc[:, 3:].values
