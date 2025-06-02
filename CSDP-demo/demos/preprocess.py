from csdp_datastore import HOMEPAP

def main():
    # Put exact filepath to the root of raw dataset (Here it it HOMEPAP). The current data is just for the demo, so download it yourself from ERDA
    raw_data_path = "O:/Tech_NTLab/DataSets/testData/sleep_data_set/csdp_demo/raw/homepap"

    # Put directory for where the HDF5 file should be saved.
    output_data_path = ""

    a = HOMEPAP(dataset_path = raw_data_path, 
                output_path = output_data_path,
                max_num_subjects = 3)

    a.port_data()

if __name__=="__main__":
    main()