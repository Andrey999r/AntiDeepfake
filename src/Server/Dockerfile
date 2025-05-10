FROM mcr.microsoft.com/dotnet/sdk:9.0-preview AS build
WORKDIR /src

COPY Services/Services.csproj ./Services/
COPY Backend/Backend.csproj ./Backend/
RUN dotnet restore ./Backend

COPY Services/. ./Services/
COPY Backend/. ./Backend/
WORKDIR /src/Backend
RUN dotnet publish -c Release -o /app/Backend

FROM mcr.microsoft.com/dotnet/aspnet:9.0-preview AS final


RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0

WORKDIR /app/Model
COPY Model/requirements.txt .

RUN python3 -m venv /app/venv
RUN /app/venv/bin/pip install --upgrade pip
RUN /app/venv/bin/pip install -r requirements.txt

COPY Model/. .

WORKDIR /app/Backend
COPY --from=build /app/Backend ./

ENTRYPOINT ["dotnet", "Backend.dll"]
