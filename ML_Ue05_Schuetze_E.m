function [] = ML_Ue05_Schuetze_Eve() 
  cars = dlmread("cars.csv", ",", 1, 1);

  #x0 = cyliner (row 1 of cars)
  x0 = initializeData(1, cars);
  #x1 = displacement (row 2 of cars)
  x1 = initializeData(2, cars);
  #x2 = horsepower (row 3 of cars)
  x2 = initializeData(3, cars);
  #x3 = weight (row 4 of cars)
  x3 = initializeData(4, cars);
  #x4 = acceleration (row 5 of cars)
  x4 = initializeData(5, cars);
  #x5 = year (row 6 of cars)
  x5 = initializeData(6, cars);
  
  #y
  y = initializeData(7, cars);
  
  rmse=100;
  
  
  x = [x0,x1,x2,x3,x4,x5];
  rounds = 100;
  #first generation
  thetaStart = ones(6,1);
  rmseMatrix = ones(rounds,1);

  mat = linearRegression(x,y,thetaStart,0.01,cars,rmseMatrix);
  mat2 = linearRegression(x,y,thetaStart,0.1,cars,rmseMatrix);
  mat3 = linearRegression(x,y,thetaStart,1.0,cars,rmseMatrix);
  mat4 = linearRegression(x,y,thetaStart,2.0,cars,rmseMatrix);
  
plot(1:rounds,mat,1:rounds,mat2,1:rounds,mat3);
legend("0.01","0.1","1.0");
xlabel("iteration");
ylabel("rmse");
endfunction

function[rMatrix] = linearRegression(x,y,theta,alpha,collection,rMatrix)
  rounds = 100;
  m = size(collection)(1);
  deltaTNorm = (x' * ((x * theta) - y)) * (alpha/m);
  
  thetaNew = theta - deltaTNorm;
  
  diff1 = x * thetaNew - y;
  yN = x * thetaNew;
  
  denormY = denormalize(y, collection(:,7));
  denormYN = denormalize(yN, collection(:,7)); 
  localrmse = sqrt(sum((denormYN .- denormY) .^ 2)/length(denormY));
  rMatrix(1,1)= localrmse;
  
  
  #put calculated theta into a copy to avoid overwriting original 
  copyTheta = thetaNew;
  #new generations
for i = 2 : rounds  
  deltaTNormLoop = (x' * ((x * copyTheta) - y)) * (alpha/m);
  thetaNewLoop = copyTheta - deltaTNormLoop;
  
  yN = x * thetaNewLoop;
  
  denormY = denormalize(y, collection(:,7));
  denormYN = denormalize(yN, collection(:,7)); 
  rmse = sqrt(sum((denormYN .- denormY) .^ 2)/length(denormY));
  #add rmse to matrix
  rMatrix(i,1)= rmse;
  #put theta into copy to start next generation
  copyTheta = thetaNewLoop;  
endfor
endfunction


function [normData] = initializeData (row, collection)
  rowData = collection(:,row);
  #dataSorted = sort(rowData');
  normData = normalize(rowData);

endfunction

function[k] = randomK()
    k = -1 + (1+1)*rand(1,6);
endfunction

function [denormA] = denormalize (a, beforeA)
  denormA = (a * (max(beforeA) - min(beforeA)) + min(beforeA));
endfunction
  
 function [normA] = normalize (a)
  
  normA = (a - min(a))/(max(a)- min(a));
endfunction