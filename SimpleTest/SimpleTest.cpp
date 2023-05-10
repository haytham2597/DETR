// SimpleTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main()
{
	std::vector<int> unassigned;
	int n = 3;
	for (int i = 1; i < n; i++)
		unassigned.push_back(i);
	for(int i=0;i<unassigned.size();i++)
	{
		std::cout << unassigned[i] << ", ";
	}

}
