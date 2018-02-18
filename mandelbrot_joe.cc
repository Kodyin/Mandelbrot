/**
 *  \file mandelbrot_susie.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>

#include "render.hh"
#include <mpi.h>

using namespace std;

int mandelbrot(double x, double y) {
	int maxit = 511;
	double cx = x;
	double cy = y;
	double newx, newy;

	int it = 0;
	for (it = 0; it < maxit && (x * x + y * y) < 4; ++it) {
		newx = x * x - y * y + cx;
		newy = 2 * x * y + cy;
		x = newx;
		y = newy;
	}
	return it;
}

int
main (int argc, char* argv[])
{
  	/* Lucky you, you get to write MPI code */
	/* Lucky me! */
	double minX = -2.1;
	double maxX = 0.7;
	double minY = -1.25;
	double maxY = 1.25;
	  
	int height, width;
	if (argc == 3) {
	 	height = atoi (argv[1]);
	    width = atoi (argv[2]);
	    assert (height > 0 && width > 0);
	} else {
	    fprintf (stderr, "usage: %s <height> <width>\n", argv[0]);
	    fprintf (stderr, "where <height> and <width> are the dimensions of the image.\n");
	    return -1;
	}

	double it = (maxY - minY)/height;
	double jt = (maxX - minX)/width;
	double x, y;

	int rank, size, offset, rowPerPro;
	double *recv;

	
	double startTime;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	rowPerPro = height / size;
	if(rank == 0) {
		startTime = MPI_Wtime();
		recv = (double *) malloc(height * width * sizeof(double));
	}
	double *send = (double *) malloc(rowPerPro * width * sizeof(double));

	y = minY + rank * rowPerPro * it;
	offset = 0;
	for (int i = 0; i < rowPerPro; i++) {
		x = minX;
		for (int j = 0; j < width; j++) {
			send[offset + j] = mandelbrot(x, y) / 512.0;
			x += jt;
		}
		offset += width;
		y += it;
	}
	offset = rank*rowPerPro;
	MPI_Barrier (MPI_COMM_WORLD);
	MPI_Gather(send, rowPerPro * width, MPI_DOUBLE, recv + offset, rowPerPro * width, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(rank == 0){
		gil::rgb8_image_t img(height, width);
		auto img_view = gil::view(img);
		for (int i = 0; i < size * rowPerPro; ++i) {
			offset = i * width;
			for (int j = 0; j < width; ++j) {
				img_view(j, i) = render(recv[j + offset]);
			}
		}
		y = minY + size * rowPerPro * it;
		for (int i = size * rowPerPro; i < height; i++) {
			x = minX;
			for (int j = 0; j < width; j++) {
				img_view(j, i) = render(mandelbrot(x, y) / 512.0);
				x += jt;
			}
			y += it;
		}
		double endTime = MPI_Wtime();
		cout <<"mandelbrot_joe"<<endl;
		cout << endTime-startTime << endl;
		
		gil::png_write_view("mandelbrot_joe.png", const_view(img));
	}



	MPI_Finalize();

}

/* eof */
