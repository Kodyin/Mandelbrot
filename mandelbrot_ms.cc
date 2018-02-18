/**
 *  \file mandelbrot_ms.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>
#include <mpi.h> 
#include "render.hh"


int
mandelbrot(double x, double y) {
	int maxit = 511;
	double cx = x;
	double cy = y;
	double newx, newy;

  	int it = 0;
  	for (it = 0; it < maxit && (x*x + y*y) < 4; ++it) {
    	newx = x*x - y*y + cx;
    	newy = 2*x*y + cy;
    	x = newx;
    	y = newy;
  	}
  	return it;
}

void slave(int rank, int width, double minX, double minY, double jt, double it) {
	int send[width + 1];

	MPI_Status status;
	int numOfRow;
	double x, y;

	while (true) {
		MPI_Recv(&numOfRow, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		//Time to terminate
		if (status.MPI_TAG == 0) return;
		else {
			x = minX;
			y = minY + (numOfRow * it);
			for (int i = 0; i < width; i++) {
				send[i] = mandelbrot(x, y);
				x += jt;
			}
			send[width] = numOfRow; 
			MPI_Send(send, width + 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
		}
	}
}

void master(int height, int width, double minX, double minY, double jt, double it, int numOfProc) {

	double startTime = MPI_Wtime();

	int *result = (int*) malloc( height * width * sizeof(int));
	MPI_Status status;
	int *recv = (int*) malloc( (width+1) * sizeof(int));
	int nextRow = 0;

	for (int i = 1; i < numOfProc; i++) {
		//start of each proc
		MPI_Send(&nextRow, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		nextRow++;
	}

	while (nextRow < height) {
		// Recv and assign new row
		MPI_Recv(recv, width + 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Send(&nextRow, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
		memcpy(result + (recv[width] * width), recv, width * sizeof(int));
		nextRow++;
	}
	
	// receive results from trailing work and terminate processes:
	for (int i = 1; i < numOfProc; i++) {
		MPI_Recv(recv, width + 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Send(0, 0, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
		// store the received result:
		memcpy(result + (recv[width] * width), recv, width * sizeof(int));
	}

	gil::rgb8_image_t img(height, width);
  	auto img_view = gil::view(img);
  	for (int k = 0; k < height; ++k) {
      	for (int p = 0; p < width; ++p) {
      		img_view(p, k) = render(result[ (k * width) + p] / 512.0); 
      	}
  	}

  	double totalT = MPI_Wtime() - startTime;
  	printf("Master & Slave  Time: %f\r\n", totalT);
  	gil::png_write_view("mandelbrot-ms.png", const_view(img));
}

int
main (int argc, char* argv[])
{
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

  	MPI_Init(&argc, &argv);
  	int numOfProc, rank;

  	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  	MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);

  	if (rank == 0) 
  	{
  		master(height, width, minX, minY, jt, it, numOfProc); 
  	} 
  	else
  	{
  		slave(rank, width, minX, minY, jt, it); 
  	}

  	MPI_Finalize();

  	return 0;

}

/* eof */
