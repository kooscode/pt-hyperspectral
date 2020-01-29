#include <iostream>
#include <utility>
#include <map>
#include <cctype>

#include <algorithm>


//LIBTERRACLEAR
#include "filetools.hpp"

//PyTorch model
#include <torch/torch.h>
#include "ann_model.hpp"

namespace tc = terraclear;

struct band_spec
{
    std::string band_name;
    uint32_t    band_index;
    uint32_t    freq_Start;
    uint32_t    freq_end;
};

struct band_val
{   
    band_spec  band_val_spec;
    uint32_t    freq_count;
    float       freq_sum;
    float       freq_avg;
};

struct class_row
{   
    std::string class_name;
    std::vector<band_val> band_values;
};

int main(int argc, char **argv) 
{
    std::cout << "Processing CSV..." << std::endl << std::flush;

    //setup band list.
    std::vector<band_spec> bands;

    // //Micasense Altum Bands
    // band_spec red = {"red", 0, 465, 485};          //475 center, 20 width
    // bands.push_back(red);
    // band_spec green = {"green", 1, 550, 570};      //560 center, 20 width
    // bands.push_back(green);
    // band_spec blue = {"blue", 2, 663, 673};        //668 center, 10 width
    // bands.push_back(blue);
    // band_spec nir = {"nir", 3, 712, 722};          //717 center, 10 width
    // bands.push_back(nir);
    // band_spec rededge = {"rededge", 4, 820, 860};  //840 center, 40 width
    // bands.push_back(rededge);

    // //Custom Bands
    band_spec b360 = {"A", 0, 459, 461}; 
    bands.push_back(b360);
    band_spec b429 = {"B", 1, 528, 530}; 
    bands.push_back(b429);
    band_spec b580 = {"C", 2, 679, 681}; 
    bands.push_back(b580);
    band_spec b839 = {"D", 3, 938, 940}; 
    bands.push_back(b839);
    band_spec b1028 = {"E", 4, 1127, 1129}; 
    bands.push_back(b1028);
    band_spec b1442 = {"F", 5, 1541, 1543}; 
    bands.push_back(b1442);
    band_spec b2279 = {"G", 6, 2378, 2380}; 
    bands.push_back(b2279);


    //which collumn contains class types
    uint32_t class_col_index = 0;
    std::string class_col_name = "labels";

    //map class types to values from file
    std::string class1 = "rock";
    std::string class2 = "notrock";
    
    std::map<std::string, std::string> map_label_class;
    map_label_class["Rock"] = class1;
    map_label_class["Soil"] = class2;
    map_label_class["Crop Residue"] = class2;

    //band to collumn index mapping
    std::map<uint32_t, uint32_t> map_band_index;

    //vector to hold data.
    std::vector<class_row> rows_all;
    std::vector<class_row> rows;
    std::vector<class_row> rows_test;

    std::string csv_file = "/data/ml/datasets/hyperspectral/terraclear/TerraClear_Spectrometer_Data.csv";
    if (!tc::filetools::file_exists(csv_file))
    {
        std::cout << "CSV not found" << std::endl;
        return -1;
    }

    //load data
    std::vector<std::string> lines = tc::filetools::read_lines(csv_file);

    //exit if no data..
    if (lines.size() <= 0)
    {
        std::cout << "No Data in CSV" << std::endl;
        return -1;
    }

    uint32_t line_num = 0;
    uint32_t col_num = 0;

    uint32_t crop_res=0;
    uint32_t soil=0;
    uint32_t rock=0;

    //iterate over all lines
    for (std::string line : lines)
    {
        std::vector<std::string> col_vals = tc::filetools::split_string(line, ',');

        //Construct Frequency index map from row 0 (header).
        if (line_num == 0)
        {
            for (std::string col_val : col_vals)
            {
                //find class name collumn..
                if (col_val.find(class_col_name) !=std::string::npos)
                {
                    class_col_index = col_num;
                }
                else
                {
                    //map frequencies to collumn index.
                    try
                    {
                        uint32_t freq = std::stoul(col_val);
                        map_band_index[freq] = col_num;
                    }
                    catch(const std::invalid_argument& e)
                    {
                        //do nothing on conversion to in error..
                    }
                }
                
                col_num++;
            }
        }
        else
        {
            //contains keyword..
            // if (class_string.find("Rock") !=std::string::npos)
            std::string class_string = col_vals.at(class_col_index);
            class_string = class_string.substr(0, class_string.length()-1); 

            //is this a class we are interrested in, if so, grab it and calc vals.
            if  (map_label_class.count(class_string) > 0)    
            {
                class_row clsrow;
                clsrow.class_name = map_label_class[class_string];

                //calc band vals
                for (auto band : bands)
                {
                    band_val bandval = { band, 0, 0, 0};
                    //grab all frequencies for this band and sum
                    for (uint32_t band_freq = band.freq_Start; band_freq != band.freq_end; band_freq++)                    
                    {   
                        //ensure we have data for this frequency
                        if (map_band_index.count(band_freq) > 0)
                        {
                            uint32_t freqindex = map_band_index[band_freq];
                            std::string freqvalstr = col_vals.at(freqindex);
                            bandval.freq_sum +=  std::stof(freqvalstr);
                            bandval.freq_count ++;
                        }
                    }

                    //calc avg
                    bandval.freq_avg = bandval.freq_sum / bandval.freq_count;
                    clsrow.band_values.push_back(bandval);
                }
                rows_all.push_back(clsrow);
                
            } //endif map class mapped            
        } //end else line-num > 0

        line_num++;
    }

    //shuffle train/test data
    std::random_shuffle ( rows_all.begin(), rows_all.end() );

    if (rows_all.size() <= 0)
    {
        std::cout << "No Data post band filtering.." << std::endl;
        return -1;
    }

    //split data..    
    float val_split = 0.10; 
    int rowcount = 0;
    for (auto row : rows_all)
    {
        if (rowcount < rows_all.size() * val_split)
            rows_test.push_back(row);
        else
            rows.push_back(row);

        rowcount++;
    }

   rows_all.clear();

    std::cout << "\tTraining_Data:" << rows.size() << ", Test_Data:" << rows_test.size() << ", Frequencies:" << map_band_index.size() << ", Bands=" << bands.size() << std::endl << std::flush;


// *****************************   PYTORCH *********************

    if (!torch::cuda::is_available())
    {
        std::cout << "CUDA not available!" << std::endl;
        return -1;        
    }

    std::cout << "Initializing Neural Network..." << std::endl << std::flush;
    std::shared_ptr<ann_model> net_ptr = std::make_shared<ann_model>();
   
    //move to GPU if CUDA available
    std::cout << "\tUploading model to GPU..." << std::endl << std::flush;
    net_ptr->to(at::kCUDA);    

    //create cpu Tensor for input data + pointer
    at::Tensor in_cpu = torch::rand({(uint32_t)bands.size(),});
    float* in_ptr = in_cpu.accessor<float, 1>().data();

    //create cpu Tensor for truth data + pointer
    at::Tensor truth_cpu = torch::rand({2,});
    float* truth_ptr = truth_cpu.accessor<float, 1>().data();

    //run through all rows for X epochs
    int batch_size = 10;
    int batch_current = 0;
    int batch_count = 0;
    float avg_loss = 0.0f;
    float epoch_max = 700;
    float epoch_map_calc = 50;
    std::string model_file = "hyperspectral.pt";

    if (tc::filetools::file_exists(model_file))
    {
        torch::load(net_ptr, model_file);
        std::cout << "\tLoaded weights from: " << model_file << std::endl;
    }
    else
    {

    std::cout << "Training..." << std::endl;
    //Create SGD optimizer with specific learning rate
    //    torch::optim::SGD optimizer(net_ptr->parameters(), /*lr=*/0.01);
    torch::optim::Adam optimizer(net_ptr->parameters(), /*lr=*/0.001);

        for (int epoch = 0; epoch < epoch_max; epoch++)
        {
            batch_count = 0;
            for (auto clsrow : rows)
            {
                // Reset gradients.
                optimizer.zero_grad();

                //set input training data
                for (int i = 0; i < in_cpu.numel(); i++)
                    in_ptr[i] = clsrow.band_values.at(i).freq_avg;

                //truth for output data one hot encoded
                truth_ptr[0] = (clsrow.class_name.compare(class1) == 0) ? 1.0f : 0.0f;
                truth_ptr[1] = (clsrow.class_name.compare(class2) == 0) ? 1.0f : 0.0f;

                //upload to GPU
                at::Tensor in_gpu = in_cpu.to(at::kCUDA);
                at::Tensor truth_gpu = truth_cpu.to(at::kCUDA);

                //forward pass
                at::Tensor out_gpu = net_ptr->forward(in_gpu);

                // copy to HOST
                at::Tensor out_cpu = out_gpu.to(at::kCPU);
                
                // Compute a loss value to judge the prediction of our model.
                torch::Tensor loss = torch::binary_cross_entropy(out_gpu, truth_gpu);

                //backward propogation of loss
                loss.backward();
                avg_loss += loss.item<float>();

                //recalc gradients.
                optimizer.step();            

            } // end rows

            avg_loss = avg_loss / rows.size();
            std::cout << "\tEpoch:" << epoch << ", Loss:" << avg_loss << std::endl << std::flush;
            avg_loss = 0.0f;

            torch::save(net_ptr, model_file);

        } //end epochs.

    } //end if file exists

    //Test accuracy
    avg_loss = 0.0f;
    batch_current = 0;

    float tp = 0;
    float fn = 0;
    float fp = 0;

    std::cout << "Testing..." << std::endl << std::flush;

    for (auto clsrow : rows_test)
    {
            //set input TEST data
            for (int i = 0; i < in_cpu.numel(); i++)
                in_ptr[i] = clsrow.band_values.at(i).freq_avg;
             
            //truth for output data one hot encoded
            truth_ptr[0] = (clsrow.class_name.compare(class1) == 0) ? 1.0f : 0.0f;
            truth_ptr[1] = (clsrow.class_name.compare(class2) == 0) ? 1.0f : 0.0f;

            //upload to GPU
            at::Tensor in_gpu = in_cpu.to(at::kCUDA);
            at::Tensor truth_gpu = truth_cpu.to(at::kCUDA);

            //forward pass
            at::Tensor out_gpu = net_ptr->forward(in_gpu);

            // copy to HOST
            at::Tensor out_cpu = out_gpu.to(at::kCPU);
            float* out_ptr = out_cpu.accessor<float, 1>().data();

            //compare output to truth...
            if (clsrow.class_name.compare(class1) == 0)
            {
                if ((out_ptr[0] > out_ptr[1]))
                    tp++;
                else 
                    fn++;
            }
            else if (clsrow.class_name.compare(class2) == 0)
            {
                if ((out_ptr[1] < out_ptr[0]))
                    fp++;
                
            }

            // Compute a loss 
            torch::Tensor loss = torch::binary_cross_entropy(out_gpu, truth_gpu);

            avg_loss += loss.item<float>();
            batch_current++;
    }

    avg_loss = avg_loss / batch_current;

    float recall = tp / (tp + fn);
    float mAP = tp / (tp + fn + fp);

    std::cout << "\tRows:" << batch_current << ", Recall:" << recall << "%, mAP:" << mAP <<"%, Loss:" << avg_loss << std::endl << std::flush;


    return 0;
}
// </code>
