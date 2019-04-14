import argparse
import os
import torchvision
import torchvision.transforms as transforms
import torch
import classify_svhn
from classify_svhn import Classifier
import numpy as np
import scipy as sp

SVHN_PATH = "svhn"
PROCESS_BATCH_SIZE = 32


def get_sample_loader(path, batch_size):
    """
    Loads data from `[path]/samples`

    - Ensure that path contains only one directory
      (This is due ot how the ImageFolder dataset loader
       works)
    - Ensure that ALL of your images are 32 x 32.
      The transform in this function will rescale it to
      32 x 32 if this is not the case.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    data = torchvision.datasets.ImageFolder(
        path,
        transform=transforms.Compose([
            transforms.Resize((32, 32), interpolation=2),
            classify_svhn.image_transform
        ])
    )
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=0,
    )
    return data_loader


def get_test_loader(batch_size):
    """
    Downloads (if it doesn't already exist) SVHN test into
    [pwd]/svhn.

    Returns an iterator over the tensors of the images
    of dimension (batch_size, 3, 32, 32)
    """
    testset = torchvision.datasets.SVHN(
        SVHN_PATH, split='test',
        download=True,
        transform=classify_svhn.image_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
    )
    return testloader


def extract_features(classifier, data_loader):
    """
    Iterator of features for each image.
    """
    with torch.no_grad():
        for x, _ in data_loader:
            h = classifier.extract_features(x).numpy()
            for i in range(h.shape[0]):
                yield h[i]


def calculate_fid_score(sample_feature_iterator,
                        testset_feature_iterator):
    """
    To be implemented by you!
    """
    #raise NotImplementedError(
    #    "TO BE IMPLEMENTED."
    #    "Part of Assignment 3 Quantitative Evaluations"
    #)

    #First loop, we pack everything together
    number_of_samples=0
    number_of_test_samples=0
    size = 0
    samples=[]
    for sample in sample_feature_iterator:
        #if number_of_samples >= 1000: break
        if (number_of_samples==0): 
            size=sample.size
            samples=np.empty((0,size))
        samples=np.append(samples,[sample], axis=0)
        number_of_samples=number_of_samples+1
    
    print("samples.shape",samples.shape)

    test_items=[]
    for test_item in testset_feature_iterator:
        #if number_of_test_samples >= 1000: break
        if (number_of_test_samples==0): 
            size=sample.size
            test_items=np.empty((0,size))
        test_items=np.append(test_items, [test_item], axis=0)
        
        number_of_test_samples=number_of_test_samples+1
    print (number_of_samples, number_of_test_samples, size)

    #Then we work on the packed values. 
    mean_samples = np.mean(samples,axis=0)
    cov_samples = np.cov(samples, rowvar=False)

    mean_test = np.mean(test_items,axis=0)
    cov_test = np.cov(test_items, rowvar=False)
    #print(mean_samples)
    #print(mean_test)
    np.savetxt("mean_samples.txt", mean_samples)
    np.savetxt("cov_samples.txt", cov_samples)
    np.savetxt("mean_test.txt", mean_test)
    np.savetxt("cov_test.txt", cov_test)

    print(mean_samples.shape)
    print(cov_samples.shape)
    print(mean_test.shape)
    print(cov_test.shape)

    #First term of the RHS of the equation
    delta_mean = mean_test-mean_samples
    norm = np.linalg.norm(delta_mean,2)
    print(norm)
    squared_norm = np.square(norm)

    #Second term of the RHS of the equation
    #print(cov_test)
    #print(cov_samples)
    cov_part = np.trace(cov_test + cov_samples - 2*sp.linalg.sqrtm(np.matmul(cov_test,cov_samples)))
    #cov_part = np.trace(cov_test + cov_samples - 2*sp.linalg.sqrtm(np.dot(cov_test,cov_samples)))
    #print (cov_part)
    print("squared_norm", squared_norm)
    print ("cov_part", cov_part)

    return cov_part + squared_norm

    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Score a directory of images with the FID score.')
    parser.add_argument('--model', type=str, default="svhn_classifier.pt",
                        help='Path to feature extraction model.')
    parser.add_argument('directory', type=str,
                        help='Path to image directory')
    args = parser.parse_args()

    quit = False
    if not os.path.isfile(args.model):
        print("Model file " + args.model + " does not exist.")
        quit = True
    if not os.path.isdir(args.directory):
        print("Directory " + args.directory + " does not exist.")
        quit = True
    if quit:
        exit()
    print("Test")
    classifier = torch.load(args.model, map_location='cpu')
    classifier.eval()
    sample_loader = get_sample_loader(args.directory,
                                      PROCESS_BATCH_SIZE)
    sample_f = extract_features(classifier, sample_loader)

    test_loader = get_test_loader(PROCESS_BATCH_SIZE)
    test_f = extract_features(classifier, test_loader)


    fid_score_unit_test = calculate_fid_score(test_f, test_f)

    print("FID score unit test", fid_score_unit_test)
    fid_score = calculate_fid_score(sample_f, test_f)
    print("FID score:", fid_score)
