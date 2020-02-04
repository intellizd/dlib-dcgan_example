#include <algorithm>
#include <iostream>
#include "stdafx.h"

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

using gray_pixel = uint8_t;
constexpr size_t noise_size = 100;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;


namespace DCGAN_CIFAR {



	noise_t make_noise()
	{
		noise_t noise;
		
		std::for_each(begin(noise), end(noise), [](matrix<float, 1, 1>& m)
			{
				m = matrix_cast<float>(randm(1, 1));
			});
		return noise;
	}

	template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
	using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;
	template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
	using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

	using generator_type =
		loss_binary_cross_entropy < fc_no_bias <1,
		htan<contp<1, 4, 2, 1,
		relu<bn_con<contp<64, 4, 2, 1,
		relu<bn_con<contp<128, 4, 2, 1,
		relu<bn_con<contp<256, 4, 2, 1, //16
		relu<bn_con<contp<512, 4, 1, 0, //16
		fc_no_bias<1024,input<noise_t>
		>>>>>>>>>>>>>>>>>;

	using discriminator_type =
		loss_binary_cross_entropy <fc<1,
		conp<1, 4, 1, 0, //1
		leakyrelu<bn_con<conp<512, 4, 2, 1, //2
		leakyrelu<bn_con<conp<256, 4, 2, 1, //4
		leakyrelu<bn_con<conp<128, 4, 2, 1, //8
		leakyrelu<conp<64, 4, 2, 1, //16
		dlib::input<dlib::matrix<gray_pixel>>
		>>>>>>>>>>>>>>;
	matrix<gray_pixel> generated_image(generator_type net)
	{
		matrix<float> output = image_plane(layer<2>(net).get_output());
		matrix<gray_pixel> image;
		assign_image_scaled(image, output);
		return image;
	}
	void rgb_image_to_grayscale_image(const dlib::matrix<dlib::rgb_pixel>& rgb_image, dlib::matrix<gray_pixel>& gray_image) {
		gray_image.set_size(rgb_image.nr(), rgb_image.nc());
		std::transform(rgb_image.begin(), rgb_image.end(), gray_image.begin(),
			[](rgb_pixel a) {return gray_pixel(a.red * 0.299f + a.green * 0.587f + a.blue * 0.114f); });
	}

	struct image_info
	{
		enum label_ label_image;
		string filename;
	};
	// A single training sample. A mini-batch comprises many of these.
	enum label_ : uint16_t
	{
		fake_image,
		real_image,
		test_image
	};
	struct training_sample
	{
		matrix<gray_pixel> real_image;
		matrix<gray_pixel> fake_image;
	};

	// ----------------------------------------------------------------------------------------


	std::vector<image_info> get_image_listing(
		const std::string& images_folder,
		const enum label_& label
	)
	{
		std::vector<image_info> results;
		image_info temp;
		temp.label_image = label;

		auto dir = directory(images_folder);
		for (auto image_file : dir.get_files())
		{
			temp.filename = image_file;
			results.push_back(temp);
		}

		return results;
	}


	int main(int argc, char** argv) try
	{
		// This example is going to run on the MNIST dataset.
		if (argc != 2)
		{
			cout << "This example needs the LSVRC 64*64 dataset to run!" << endl;
			cout << "You can get LSVRC from http://www.image-net.org/challenges/LSVRC/" << endl;
			cout << "Download the files that comprise the dataset, decompress them, and" << endl;
			cout << "put them in a folder train-dcgan\\ILSVR\\64. You can put anything(size 64*64) in the folder." << end;
			cout <<" Then give that folder as input to this program." << endl;

			return EXIT_FAILURE;
		}

		dcganCIFAR(argv[1], 16, 4, false, 0.8, 1, 1, 1, true);
	}


std::vector<image_info> get_train_listing_real(const std::string& ifolder) { return get_image_listing(ifolder, real_image); }
std::vector<image_info> get_train_listing_fake(const std::string& ifolder) { return get_image_listing(ifolder, fake_image); }
std::vector<image_info> get_train_listing_test(const std::string& ifolder) { return get_image_listing(ifolder, test_image); }
 	int dcganCIFAR(const char* TrainPath, int minibatch, int geninterval, bool random_noise, float realLow, float realHigh, float fakeLow, float fakeHigh, bool bTrainFake)		try
	{
		// This example is going to run on the MNIST dataset.

		bool bGen = false;
		bool bRealSave = false;


		generator_type generator;
		discriminator_type discriminator(
			leakyrelu_(0.2), leakyrelu_(0.2), leakyrelu_(0.2), leakyrelu_(0.2));
		cout << "generator" << endl;
		cout << generator << endl;
		cout << "discriminator" << endl;
		cout << discriminator << endl;

		
		dnn_trainer<generator_type, adam> gen_trainer(generator, adam(0, 0.9, 0.99999));
		gen_trainer.set_synchronization_file("dcgan_CIFAR_generator_sync", std::chrono::seconds(600));
		gen_trainer.be_verbose();
		gen_trainer.set_learning_rate(2e-4);
		gen_trainer.set_learning_rate_shrink_factor(0.9);
		gen_trainer.set_iterations_without_progress_threshold(2000);
		cout << gen_trainer << endl;
		dnn_trainer<discriminator_type, adam> dis_trainer(discriminator, adam(0, 0.9, 0.99999));
		dis_trainer.set_synchronization_file("dcgan_CIFAR_discriminator_sync", std::chrono::seconds(600));
		dis_trainer.be_verbose();
		dis_trainer.set_learning_rate(2e-4);
		dis_trainer.set_learning_rate_shrink_factor(0.9); 
		dis_trainer.set_iterations_without_progress_threshold(2000);
		cout << dis_trainer << endl;
		double g_loss = 0;
		const long minibatch_size = minibatch;
		dis_trainer.set_mini_batch_size(minibatch);
		gen_trainer.set_mini_batch_size(minibatch);
		dlib::rand rnd(time(nullptr));
		std::vector<noise_t> noises;
		std::vector<noise_t> fixed_noises;

	
			for (int i = 0; i < minibatch_size; i++)
			{
				noise_t noise = make_noise();
				noises.push_back(noise);
			}
			noises.clear();
			for (int i = 0; i < minibatch_size; i++)
			{
				noise_t noise = make_noise();
				noises.push_back(noise);
			}

	
		for (int i = 0; i < minibatch_size; i++)
		{
			noise_t noise = make_noise();
			fixed_noises.push_back(noise);
		}
		int trainCnt = 0;
		auto real_images = get_train_listing_real("train-dcgan\\ILSVR\\64");
		cout << "real examples: " << real_images.size() << endl;


		const unsigned long image_size = 64; 
	

		std::vector<matrix<gray_pixel>> mini_batch_fake_samples;
		std::vector<matrix<gray_pixel>> mini_batch_gen_make_samples;
		std::vector<float> mini_batch_fake_labels;

		std::vector<matrix<gray_pixel>> mini_batch_real_samples;
		std::vector<float> mini_batch_real_labels;
		training_sample sample;

		string real_filename = dlib::get_current_dir() + string("/real_image_generator_state.jpg");
		string fake_filename = dlib::get_current_dir() + string("/fake_image_generator_state.jpg");

		while (gen_trainer.get_train_one_step_calls() < 1000)
		{
			// train the discriminator with real images

			int idx = 0;

			while (mini_batch_real_samples.size() < minibatch_size)
			{

				image_info temp;
				matrix<rgb_pixel> in, out;
				temp = real_images[idx];
				load_image(in, temp.filename);
				training_sample sample;
				rgb_image_to_grayscale_image(in, sample.real_image);		
				mini_batch_real_samples.push_back(sample.real_image);


					idx++;
			}
			mini_batch_real_labels.clear();
			while (mini_batch_real_labels.size() < minibatch_size)
			{
				mini_batch_real_labels.push_back(rnd.get_double_in_range(realLow, realHigh));
			}
			dis_trainer.train_one_step(mini_batch_real_samples, mini_batch_real_labels);

			/*if (random_noise)
			{
				noises.clear();

				for (int i = 0; i < minibatch_size; i++)
				{
					noise_t noise = make_noise();
					noises.push_back(noise);
				}
			}*/
			int labelcnt = 0;
			if (bGen == false) {
				while (mini_batch_fake_samples.size() < minibatch_size)
				{

					generator(noises[labelcnt]);
					matrix<gray_pixel> fake_img = generated_image(generator);
					mini_batch_fake_samples.push_back(fake_img);
					mini_batch_fake_labels.push_back(rnd.get_double_in_range(0, 0));
				//	dlib::save_png(fake_img, std::to_string(trainCnt / 1000) + "_img_" + std::to_string(labelcnt) + "_F.png");
					labelcnt++;
					bGen = true;
				}
			}
			/*const resizable_tensor& out_fake2 = discriminator.subnet().subnet().get_final_data_gradient();
			std::cout << "out_fake2: " <<
				out_fake2.num_samples() << "x" <<
				out_fake2.k() << "x" <<
				out_fake2.nr() << "x" <<
				out_fake2.nc() << std::endl;

			const resizable_tensor& out_gen = generator.subnet().subnet().get_output();
			std::cout << "gen_fake:  " <<
				out_gen.num_samples() << "x" <<
				out_gen.k() << "x" <<
				out_gen.nr() << "x" <<
				out_gen.nc() << std::endl;*/
			dis_trainer.train_one_step(mini_batch_fake_samples, mini_batch_fake_labels);

			if (bTrainFake)
			{
				mini_batch_fake_labels.clear();
				while (mini_batch_fake_labels.size() < minibatch_size)
				{

					mini_batch_fake_labels.push_back(rnd.get_double_in_range(1, 1));

				}

				dis_trainer.test_one_step(mini_batch_fake_samples, mini_batch_fake_labels);
			}
			resizable_tensor noises_tensor;
			layer<17>(generator).to_tensor(noises.begin(), noises.end(), noises_tensor);
			generator(noises);
			//const resizable_tensor& out_fake = discriminator.subnet().subnet().get_final_data_gradient();
			const resizable_tensor& out_fake = discriminator.subnet().subnet().get_final_data_gradient();

			if (out_fake.nr() > 0 && out_fake.nc() > 0)
			{
				auto solvers = gen_trainer.get_solvers();

				make_sstack<adam>(solvers);



				generator.subnet().subnet().back_propagate_error(noises_tensor, out_fake);
				generator.subnet().subnet().update_parameters(make_sstack<adam>(solvers), gen_trainer.get_learning_rate());

				



			}
			else
			{
				std::cout << "Empty out_fake, skipping..." << std::endl;
			}





			if (trainCnt % geninterval == 0)
			{
				mini_batch_fake_samples.clear();
				mini_batch_fake_labels.clear();
				int labelcnt = 0;

				while (mini_batch_fake_samples.size() < minibatch_size)
				{
					generator(noises[labelcnt]);
					matrix<gray_pixel> fake_img = generated_image(generator);
					mini_batch_fake_samples.push_back(fake_img);
					mini_batch_fake_labels.push_back(rnd.get_double_in_range(0, 0));
					labelcnt++;

				}


			}
			if (trainCnt % 50 == 0)
			{
				for (int i = 0; i < minibatch_size; i++)
				{
				
				if(random_noise)
					generator(make_noise());
				else
					generator(noises[i]);
						matrix<gray_pixel> fake_img = generated_image(generator);
						dlib::save_png(fake_img, "gen\\" + std::to_string(trainCnt / 500) + "_img_" + std::to_string(i) + "_F.png");				

					if (bRealSave == false)
						dlib::save_png(mini_batch_real_samples[i], "gen\\img_" + std::to_string(i) + "_T.png");



				}
				if (bRealSave == false)
					bRealSave = true;
			}

			trainCnt++;
			gen_trainer.test_one_step(noises, mini_batch_fake_labels);
			gen_trainer.setUpdateLastSync(true);
			
		}

		return EXIT_SUCCESS;
	}
	catch (exception & e)
	{
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}
}
