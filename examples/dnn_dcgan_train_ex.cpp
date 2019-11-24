#include <algorithm>
#include <iostream>
#include "stdafx.h"

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

constexpr size_t noise_size = 100;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;
namespace DCGAN_MNIST {

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
		relu<bn_con<contp<128, 3, 2, 1,
		relu<bn_con<contp<256, 4, 1, 0,
		input<noise_t>
		>>>>>>>>>>>>>;
	using discriminator_type =
		loss_binary_cross_entropy <fc_no_bias<1,
		conp<1, 3, 1, 0,
		leakyrelu<bn_con<conp<256, 4, 2, 1,
		leakyrelu<bn_con<conp<128, 4, 2, 1,
		leakyrelu<conp<64, 4, 2, 1,
		input<matrix<unsigned char>>
		>>>>>>>>>>>;
	matrix<unsigned char> generated_image(generator_type net)
	{
		matrix<float> output = image_plane(layer<2>(net).get_output());
		matrix<unsigned char> image;
		assign_image_scaled(image, output);
		return image;
	}
	int main()
	{

		dcGanTrain("D:\\Learning\\DCGAN\\train-images-idx3-ubyte", 16, 4, false, 0.8, 1, 1, 1, true);
	}

 
	int dcGanTrain(const char* TrainPath, int minibatch, int geninterval, bool random_noise, float realLow, float realHigh, float fakeLow, float fakeHigh, bool bTrainFake)
		try
	{
		// This example is going to run on the MNIST dataset.

		bool bGen = false;
		bool bRealSave = false;
		// MNIST is broken into two parts, a training set of 60000 images and a test set of
		// 10000 images.  Each image is labeled so that we know what hand written digit is
		// depicted.  These next statements load the dataset into memory.
		std::vector<matrix<unsigned char>> training_images;
		std::vector<unsigned long>         training_labels;
		std::vector<matrix<unsigned char>> testing_images;
		std::vector<unsigned long>         testing_labels;
		load_mnist_dataset(TrainPath, training_images, training_labels, testing_images, testing_labels);


		generator_type generator;
		discriminator_type discriminator(
			leakyrelu_(0.2), leakyrelu_(0.2), leakyrelu_(0.2));
		cout << "generator" << endl;
		cout << generator << endl;
		cout << "discriminator" << endl;
		cout << discriminator << endl;

		dnn_trainer<generator_type, adam> gen_trainer(generator, adam(0, 0.5, 0.999));
		gen_trainer.set_synchronization_file("dcgan_generator9_sync", std::chrono::hours(5));
		gen_trainer.be_verbose();
		gen_trainer.set_learning_rate(2e-4);
		gen_trainer.set_learning_rate_shrink_factor(1);
		cout << gen_trainer << endl;
		dnn_trainer<discriminator_type, adam> dis_trainer(discriminator, adam(0, 0.5, 0.9999));
		dis_trainer.set_synchronization_file("dcgan_discrimi9nator_sync", std::chrono::seconds(180));
		dis_trainer.be_verbose();
		dis_trainer.set_learning_rate(2e-4);
		dis_trainer.set_learning_rate_shrink_factor(1);
		cout << dis_trainer << endl;
		double g_loss = 0;
		const long minibatch_size = minibatch;
		dis_trainer.set_mini_batch_size(minibatch);
		gen_trainer.set_mini_batch_size(minibatch);
		dlib::rand rnd(time(nullptr));
		std::vector<noise_t> noises;

		if (random_noise == false)
		{
			for (int i = 0; i < minibatch_size; i++)
			{
				noise_t noise = make_noise();
				noises.push_back(noise);
			}
		}
		int trainCnt = 0;
		
		std::vector<matrix<unsigned char>> mini_batch_fake_samples;
		std::vector<matrix<unsigned char>> mini_batch_gen_make_samples;
		std::vector<float> mini_batch_fake_labels;
		while (gen_trainer.get_train_one_step_calls() < 1000)
		{
			// train the discriminator with real images
			std::vector<matrix<unsigned char>> mini_batch_real_samples;
			std::vector<float> mini_batch_real_labels;

			int idx = 0;
			while (mini_batch_real_samples.size() < minibatch_size)
			{
				if (random_noise)
					idx = rnd.get_random_32bit_number() % minibatch_size * 10;

				mini_batch_real_samples.push_back(training_images[idx]);
				mini_batch_real_labels.push_back(rnd.get_double_in_range(realLow, realHigh));
				if (!random_noise)
					idx++;
			}

			dis_trainer.train_one_step(mini_batch_real_samples, mini_batch_real_labels);

			if (random_noise)
			{
				noises.clear();

				for (int i = 0; i < minibatch_size; i++)
				{
					noise_t noise = make_noise();
					noises.push_back(noise);
				}
			}
			int labelcnt = 0;
			if (bGen == false) {
				while (mini_batch_fake_samples.size() < minibatch_size)
				{

					generator(noises[labelcnt]);
					matrix<unsigned char> fake_img = generated_image(generator);
					mini_batch_fake_samples.push_back(fake_img);
					mini_batch_fake_labels.push_back(rnd.get_double_in_range(0, 0));
					labelcnt++;
					bGen = true;
				}
			}
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
			layer<13>(generator).to_tensor(noises.begin(), noises.end(), noises_tensor);
			generator(noises);
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
					matrix<unsigned char> fake_img = generated_image(generator);

					mini_batch_fake_samples.push_back(fake_img);
					mini_batch_fake_labels.push_back(rnd.get_double_in_range(0, 0));
					labelcnt++;

				}

				discriminator.subnet().zero_grad();
				generator.subnet().zero_grad();


			}
			if (trainCnt % 50 == 0)
			{
				for (int i = 0; i < minibatch_size; i++)
				{
					dlib::save_png(mini_batch_fake_samples[i], std::to_string(trainCnt) + "_img_" + std::to_string(i) + "_F.png");


					if (bRealSave == false)
						dlib::save_png(mini_batch_real_samples[i], "img_" + std::to_string(i) + "_T.png");



				}
				if (bRealSave == false)
					bRealSave = true;
			}

			trainCnt++;

		}

		return EXIT_SUCCESS;
	}
	catch (exception & e)
	{
		cout << e.what() << endl;
		return EXIT_FAILURE;
	}
}