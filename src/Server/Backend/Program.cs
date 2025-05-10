using Services;
using Services.Interfaces;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddScoped<ICheckFileService, CheckFileService>();
builder.Services.AddScoped<ISaveFileService, SaveFileService>();
builder.Services.AddScoped<IRunModelService, RunModelService>();
builder.Services.AddCors();

var app = builder.Build();
app.UseCors(builder => builder.AllowAnyOrigin());

app.MapPost("/upload", async (IFormFile file, ICheckFileService checkFileService, 
    ISaveFileService saveFileService, IRunModelService runModelService) =>
{
    if (file == null) return Results.BadRequest("Файл не предоставлен");

    if (!await checkFileService.IsVideoFileAsync(file)) return Results.BadRequest("Файл не является видео");

    string filePath = await saveFileService.Save(file);

    int exitCode = await runModelService.Run(filePath);

    if (exitCode == 1) return Results.Ok("Видео DeepFake");
    if (exitCode == 2) return Results.BadRequest("Ошибка запуска");
    if (exitCode == 3) return Results.BadRequest("Файл видео не найден");

    return Results.Ok("Видео реальное");
}).DisableAntiforgery();

app.Run();

