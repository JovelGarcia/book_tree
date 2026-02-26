import { Component, OnInit, signal, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

interface EpubFile {
  id: number;
  original_filename: string;
  uploaded_at: string;
  status: 'p' | 'pr' | 'c' | 'f';
}

const STATUS_LABELS: Record<string, string> = {
  p:  'pending',
  pr: 'processing',
  c:  'done',
  f:  'failed',
};

@Component({
  selector: 'app-epub-list',
  imports: [CommonModule],
  templateUrl: './epub-list.html',
  styleUrl: './epub-list.css'
})
export class EpubList implements OnInit {
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;

  epubs        = signal<EpubFile[]>([]);
  loading      = signal(true);
  error        = signal<string | null>(null);

  panelOpen    = signal(false);
  dragOver     = signal(false);
  selectedFile = signal<File | null>(null);
  uploading    = signal(false);
  uploadError  = signal<string | null>(null);

  private readonly API = '/api/epubs/';

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.fetchEpubs();
  }

  fetchEpubs() {
    this.loading.set(true);
    this.http.get<EpubFile[]>(this.API).subscribe({
      next:  data => { this.epubs.set(data); this.loading.set(false); },
      error: ()   => { this.error.set('Could not reach the API.'); this.loading.set(false); },
    });
  }

  togglePanel() {
    this.panelOpen.update(v => !v);
    if (!this.panelOpen()) this.resetUpload();
  }

  onDragOver(e: DragEvent) {
    e.preventDefault();
    this.dragOver.set(true);
  }

  onDragLeave() {
    this.dragOver.set(false);
  }

  onDrop(e: DragEvent) {
    e.preventDefault();
    this.dragOver.set(false);
    const file = e.dataTransfer?.files[0];
    if (file) this.setFile(file);
  }

  onFileSelected(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) this.setFile(file);
  }

  setFile(file: File) {
    this.uploadError.set(null);
    if (!file.name.endsWith('.epub')) {
      this.uploadError.set('Only .epub files are accepted.');
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      this.uploadError.set('File must be under 50MB.');
      return;
    }
    this.selectedFile.set(file);
  }

  submit() {
    const file = this.selectedFile();
    if (!file) return;

    const form = new FormData();
    form.append('file', file);

    this.uploading.set(true);
    this.uploadError.set(null);

    this.http.post(`${this.API}upload/`, form).subscribe({
      next: () => {
        this.panelOpen.set(false);
        this.resetUpload();
        this.fetchEpubs();
      },
      error: (err) => {
        this.uploadError.set(err.error?.error ?? 'Upload failed. Please try again.');
        this.uploading.set(false);
      },
    });
  }

  resetUpload() {
    this.selectedFile.set(null);
    this.uploadError.set(null);
    this.uploading.set(false);
    this.dragOver.set(false);
    if (this.fileInput?.nativeElement) {
      this.fileInput.nativeElement.value = '';
    }
  }

  stripExtension(name: string): string {
    return name.replace(/\.[^.]+$/, '');
  }

  statusLabel(s: string): string {
    return STATUS_LABELS[s] ?? s;
  }
}
