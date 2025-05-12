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
    if (file == null)
        return Results.Text("Файл не предоставлен", "text/plain");

    if (!await checkFileService.IsVideoFileAsync(file))
        return Results.Text("Файл не является видео", "text/plain");

    string filePath = await saveFileService.Save(file);
    int exitCode = await runModelService.Run(filePath);

    return exitCode switch
    {
        0 => Results.Text("Видео реальное", "text/plain"),
        1 => Results.Text("Видео DeepFake", "text/plain"),
        2 => Results.Text("Ошибка запуска", "text/plain"),
        _ => Results.Text("Файл видео не найден", "text/plain")
    };
}).DisableAntiforgery();

app.Run();

