# HippoNet-cardiac
Code from Hippo-Net cardiac network (MIOT)
DEEP LEARNING STAGING OF CARDIAC IRON OVERLOAD FROM MULTI-ECHO MR IMAGES


Abstract
Purpose: To develop a deep-learning model for unsupervised classification of myocardial iron overload (MIO) from magnitude T2* multi-echo MR images.
Materials and Methods: Cardiac magnitude T2* multi-echo MR images from 823 thalassemia major patients (466 females, 56%), labelled for myocardial iron overload level, were retrospectively studied. Two 2D CNNs (MS-HippoNet and SS-HippoNet) were trained using 5-fold cross-validation. Performance was assessed using multi-class and single-class accuracy, sensitivity, and specificity. CNN performance was compared with inter-observer agreement between radiologists on 20% of the patients. The agreement between patients' classifications was assessed by the inter-agreement K test.
Results: Across the 165 patients in the test set, a multi-class accuracy of 0.885 and 0.836 was obtained for MS- and SS-Hippo-Net, respectively. Network performances were confirmed on external test set analysis (0.827 and 0.793 multi-class accuracy, 29 patients from the CHMMOTv1 database). The agreement between automatic and ground truth classification was good (MS: K=0.771; SS: K=0.614), comparable with the inter-observer agreement for the MS network (K=0.872) and SS network (K=0.907). 
Conclusion: 2D-CNNs performed classification of myocardial iron overload level from multiecho, bright-blood, T2* images.
