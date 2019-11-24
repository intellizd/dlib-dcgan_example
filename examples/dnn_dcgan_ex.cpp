#include <algorithm>
#include <iostream>

#include <dlib/data_io.h>
#include <dlib/dnn.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;

// some helper definitions for the noise generation
constexpr size_t noise_size = 100;
using noise_t = std::array<matrix<float, 1, 1>, noise_size>;

noise_t make_noise()
{
    noise_t noise;
    std::for_each(begin(noise), end(noise), [] (matrix<float, 1, 1> &m)
        {
            m = matrix_cast<float>(randm(1, 1));
        });
    return noise;
}

// a custom convolution definition to allow for padding size specification
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using conp = add_layer<con_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// Let's define a transposed convolution to with custom padding
template<long num_filters, long kernel_size, int stride, int padding, typename SUBNET>
using contp = add_layer<cont_<num_filters, kernel_size, kernel_size, stride, stride, padding, padding>, SUBNET>;

// The generator is made of a bunch of deconvolutional layers. It's input is a
// 1 x 1 x k noise tensor, and the output is the score of the generated image
// (decided by the discriminator, which we'll define afterwards)
using generator_type =
    loss_binary_log<fc<1,
    htan<contp<1, 4, 2, 1,
    relu<bn_con<contp<64, 4, 2, 1,
    relu<bn_con<contp<128, 3, 2, 1,
    relu<bn_con<contp<256, 4, 1, 0,
    input<noise_t>
    >>>>>>>>>>>>>;

// Now, let's proceed to define the discriminator, whose role will be to decide
// whether an image is fake or not.
using discriminator_type =
    loss_binary_log<fc<1,
    bn_con<conp<1, 3, 1, 0,
    prelu<bn_con<conp<256, 4, 2, 1,
    prelu<bn_con<conp<128, 4, 2, 1,
    prelu<conp<64, 4, 2, 1,
    input<matrix<unsigned char>>
    >>>>>>>>>>>>;

// Now, let's define a way to easily get the generated image
matrix<unsigned char> generated_image(generator_type net)
{
    matrix<float> output = image_plane(layer<2>(net).get_output());
    matrix<unsigned char> image;
    assign_image_scaled(image, output);
    return image;
}

int main(int argc, char** argv) try
{
    // This example is going to run on the MNIST dataset.
    if (argc != 2)
    {
        cout << "This example needs the MNIST dataset to run!" << endl;
        cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        return EXIT_FAILURE;
    }

    // MNIST is broken into two parts, a training set of 60000 images and a test set of
    // 10000 images.  Each image is labeled so that we know what hand written digit is
    // depicted.  These next statements load the dataset into memory.
    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    generator_type generator;
    discriminator_type discriminator(
            prelu_(0.2), prelu_(0.2), prelu_(0.2));
    cout << "generator" << endl;
    cout << generator << endl;
    cout << "discriminator" << endl;
    cout << discriminator << endl;

    dnn_trainer<generator_type, adam> gen_trainer(generator, adam(0, 0.5, 0.999));
    gen_trainer.set_synchronization_file("dcgan_generator_sync", std::chrono::hours(5));
    gen_trainer.be_verbose();
    gen_trainer.set_learning_rate(2e-4);
    gen_trainer.set_learning_rate_shrink_factor(1);
    cout << gen_trainer << endl;
    dnn_trainer<discriminator_type, adam> dis_trainer(discriminator, adam(0, 0.5, 0.999));
    dis_trainer.set_synchronization_file("dcgan_discriminator_sync", std::chrono::minutes(1));
    dis_trainer.be_verbose();
    dis_trainer.set_learning_rate(2e-4);
    dis_trainer.set_learning_rate_shrink_factor(1);
    cout << dis_trainer << endl;

    const long minibatch_size = 8;
    dlib::rand rnd(time(nullptr));
    while (gen_trainer.get_train_one_step_calls() < 1000)
    {
        // train the discriminator with real images
        std::vector<matrix<unsigned char>> mini_batch_real_samples;
        std::vector<float> mini_batch_real_labels;
        while (mini_batch_real_samples.size() < minibatch_size)
        {
            auto idx = rnd.get_random_32bit_number() % training_images.size();
            mini_batch_real_samples.push_back(training_images[idx]);
            mini_batch_real_labels.push_back(rnd.get_double_in_range(0.8, 1.0));
        }
        dis_trainer.train_one_step(mini_batch_real_samples, mini_batch_real_labels);

        // train the discriminator with fake images
        std::vector<matrix<unsigned char>> mini_batch_fake_samples;
        std::vector<float> mini_batch_fake_labels;
        std::vector<noise_t> noises;
        while (mini_batch_fake_samples.size() < minibatch_size)
        {
            auto noise = make_noise();
            noises.push_back(noise);
            generator(noise);
            matrix<unsigned char> fake_img = generated_image(generator);
            mini_batch_fake_samples.push_back(fake_img);
            mini_batch_fake_labels.push_back(-1.0f);
        }
        dis_trainer.train_one_step(mini_batch_fake_samples, mini_batch_fake_labels);
        const resizable_tensor& out_fake = discriminator.subnet().subnet().get_final_data_gradient();

        // std::cout << "out_fake: " <<
        //     out_fake.num_samples() << "x" <<
        //     out_fake.k() << "x" <<
        //     out_fake.nr() << "x" <<
        //     out_fake.nc() << std::endl;

        resizable_tensor noises_tensor;
        layer<13>(generator).to_tensor(noises.begin(), noises.end(), noises_tensor);
        generator(noises);

        // const resizable_tensor& out_gen = generator.subnet().subnet().get_output();
        // std::cout << "out_gen:  " <<
        //     out_gen.num_samples() << "x" <<
        //     out_gen.k() << "x" <<
        //     out_gen.nr() << "x" <<
        //     out_gen.nc() << std::endl;

        if (out_fake.nr() > 0 && out_fake.nc() > 0)
        {
            generator.subnet().subnet().back_propagate_error(noises_tensor, out_fake);
        }
        else
        {
            std::cout << "Empty out_fake, skipping..." << std::endl;
        }


        auto iteration = dis_trainer.get_train_one_step_calls();
        if (iteration % 500 == 0)
        {
            auto gen_img = generated_image(generator);
            dlib::save_png(gen_img, "img" + std::to_string(iteration) + ".png");
        }

    }

    return EXIT_SUCCESS;
}
catch(exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
