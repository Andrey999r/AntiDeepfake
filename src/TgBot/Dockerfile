FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build

WORKDIR /app

COPY TgBot/*.csproj ./TgBot/

RUN dotnet restore TgBot/TgBot.csproj

COPY . ./

RUN dotnet publish TgBot/TgBot.csproj -c Release -o /app/publish

FROM mcr.microsoft.com/dotnet/runtime:9.0 AS base

WORKDIR /app

COPY --from=build /app/publish .

ENTRYPOINT ["dotnet", "TgBot.dll"]
