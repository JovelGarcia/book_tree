import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './upload.html',
  styleUrl: './upload.css'
})

export class Upload {
  selectedFile: File | null = null;
  uploading = false;
  uploadProgress = 0;
  dragOver = false;
  errors: string[] = [];
  successMessage: string | null = null;

  readonly MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

  constructor(
    private http: HttpClient,
    private router: Router
  ) {}

  onFileSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      this.validateAndSetFile(input.files[0]);
    }
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.dragOver = true;
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.dragOver = false;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    this.dragOver = false;

    if (event.dataTransfer?.files && event.dataTransfer.files.length > 0) {
      this.validateAndSetFile(event.dataTransfer.files[0]);
    }
  }

  validateAndSetFile(file: File) {
    this.errors = [];
    this.selectedFile = null;
    this.successMessage = null;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.epub')) {
      this.errors.push('Only .epub files are accepted');
      return;
    }

    // Validate file size
    if (file.size > this.MAX_FILE_SIZE) {
      this.errors.push(`File size must be less than 50MB. Current size: ${this.formatFileSize(file.size)}`);
      return;
    }

    this.selectedFile = file;
  }

  uploadFile() {
    if (!this.selectedFile) {
      this.errors = ['Please select a file'];
      return;
    }

    this.uploading = true;
    this.uploadProgress = 0;
    this.errors = [];

    const formData = new FormData();
    formData.append('file', this.selectedFile);

    // Simulate progress (in real app with progress events)
    const progressInterval = setInterval(() => {
      if (this.uploadProgress < 90) {
        this.uploadProgress += 10;
      }
    }, 200);

    this.http.post('http://127.0.0.1:8000/api/epubs/', formData)
      .subscribe({
        next: (response) => {
          clearInterval(progressInterval);
          this.uploadProgress = 100;

          setTimeout(() => {
            this.uploading = false;
            this.successMessage = 'File uploaded successfully!';
            setTimeout(() => {
              this.router.navigate(['/']);
            }, 1500);
          }, 500);
        },
        error: (err) => {
          clearInterval(progressInterval);
          this.uploading = false;
          this.uploadProgress = 0;

          const errorMessage = err.error?.message || err.error?.detail || 'Failed to upload file';
          this.errors = [errorMessage];
          console.error(err);
        }
      });
  }

  clearFile() {
    this.selectedFile = null;
    this.errors = [];
    this.successMessage = null;
  }

  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  }
}
